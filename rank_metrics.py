"""逐 rank 指标与缓存验收；由统一 HTTP Driver 调用，不是启动入口。"""
import json
import math
import re
import time

FORWARD = {
    ('sglang:forward_execution_seconds_total', 'category', 'extend'): 'seconds',
    ('sglang:cuda_graph_passes_total', 'mode', 'prefill_none'): 'passes',
    ('sglang:realtime_tokens_total', 'mode', 'prefill_compute'): 'compute',
    ('sglang:prefill_effective_tokens_total', 'mode', 'device_hit'): 'device',
    ('sglang:prefill_effective_tokens_total', 'mode', 'host_hit'): 'host',
    ('sglang:prefill_effective_tokens_total', 'mode', 'storage_hit'): 'storage',
    ('sglang:prefill_effective_tokens_total', 'mode', 'input'): 'input',
}
LOAD = {
    'sglang:load_back_duration_seconds_sum': 'load_seconds',
    'sglang:load_back_duration_seconds_count': 'load_ops',
    'sglang:load_back_bytes_total': 'load_bytes',
}


class PendingMetrics(RuntimeError):
    """计数尚未到齐，可以继续等；拓扑错误、超量或倒退则直接失败。"""


def rank_states(state, tp):
    ranks = state.get('ranks', {'0': state} if tp == 1 else {})
    if set(ranks) != {str(r) for r in range(tp)}:
        raise RuntimeError('缓存控制返回的 rank 不完整')
    return {int(r): value for r, value in ranks.items()}


def check_state(state, group, p, tp):
    """检查每张卡的完整 P 检查点；GPU KV 命中不能代替线性状态检查。"""
    for rank, value in rank_states(state, tp).items():
        wanted = dict(device_tokens=p if group == 'l1' else 0,
                      host_hit_tokens=p if group == 'l2' else 0,
                      host_backed_tokens=p if group != 'cold' else 0)
        if 'mamba' in value.get('cache_components', []):
            wanted.update(mamba_device_states=int(group == 'l1'),
                          mamba_host_states=int(group != 'cold'))
        if group != 'cold':
            wanted['backup_complete'] = True
        for key, expected in wanted.items():
            if value.get(key) != expected:
                raise RuntimeError(f'rank={rank} {group} {key}: {value.get(key)} != {expected}')


def parse_metrics(text, tp, ep, hybrid, *, gpu_timing=True):
    """一套解析规则覆盖单卡/多卡；未启用计时时只要求 H2D 指标。

    Prometheus 全文只在内存中筛选。缺失值不能填零，重复 label 不能累加，
    否则可能把其他模型或 rank 的活动误算作本次请求。
    """
    result = {rank: {} for rank in range(tp)}
    required = set(LOAD.values()) | {'kv_tokens'}
    if gpu_timing:
        required.update(FORWARD.values())
    if hybrid:
        required.add('mamba_states')
    for line in text.splitlines():
        match = re.fullmatch(r'([\w:]+)\{(.*)\}\s+(\S+)(?:\s+\S+)?', line)
        if not match:
            continue
        name = match[1]
        labels = dict(re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[2]))
        keys = [v for (n, label, value), v in FORWARD.items() if n == name and labels.get(label) == value]
        forward = bool(keys)
        key = keys[0] if keys else LOAD.get(name)
        if name == 'sglang:load_back_tokens_total':
            key = {'kv': 'kv_tokens', 'mamba': 'mamba_states'}.get(labels.get('pool'))
        if key not in required:
            continue
        rank = int(labels.get('tp_rank', '-1'))
        if rank not in result or labels.get('pp_rank') != '0' or labels.get('dp_rank', '0') != '0':
            raise RuntimeError(f'意外指标拓扑：{labels}')
        expected_ep = str(rank if ep == tp else 0)
        if forward and labels.get('moe_ep_rank') != expected_ep:
            raise RuntimeError(f'EP 指标拓扑错误：{labels}')
        value = float(match[3])
        if key in result[rank] or not math.isfinite(value) or value < 0:
            raise RuntimeError(f'重复或非法指标 rank={rank} {key}')
        if not key.endswith('seconds') and not value.is_integer():
            raise RuntimeError(f'非整数计数 rank={rank} {key}')
        result[rank][key] = value
    missing = {r: sorted(required - v.keys()) for r, v in result.items() if required != v.keys()}
    if missing:
        raise PendingMetrics(f'指标尚未完整发布：{missing}')
    return result


def metric_delta(before, after, *, p, s, group, chunk, state, gpu_timing=True):
    """只接受与本次输入、缓存层级和实际池大小一致的累计差值。

    服务必须独占且单请求；计数差值本身没有 request ID。Cold 计算 P+S，
    命中后计算 S；各 rank 的逻辑 token 应相同，物理传输量分别检查。
    """
    compute = p + s if group == 'cold' else s
    expected = dict(kv_tokens=p if group == 'l2' else 0)
    if gpu_timing:
        expected.update(passes=math.ceil(compute / chunk), compute=compute, input=compute,
                        device=p if group == 'l1' else 0, host=p if group == 'l2' else 0, storage=0)
    deltas, pending = {}, []
    states = rank_states(state, len(before))
    if set(before) != set(after):
        raise RuntimeError('请求前后 rank 不一致')
    for rank in before:
        delta = {k: after[rank][k] - v for k, v in before[rank].items()}
        deltas[rank] = delta
        if any(v < 0 for v in delta.values()):
            raise RuntimeError(f'rank={rank} 累计指标倒退')
        for key, wanted in expected.items():
            if delta[key] > wanted:
                raise RuntimeError(f'rank={rank} {key}: {delta[key]} > {wanted}')
            if delta[key] < wanted:
                pending.append(f'{rank}:{key}')
        if gpu_timing and delta['seconds'] <= 0:
            pending.append(f'{rank}:seconds')
        if group == 'l2':
            copies = delta.get('mamba_states', 0)
            if 'mamba_states' in delta and copies not in (1, 2):
                if copies > 2:
                    raise RuntimeError('线性状态恢复次数异常')
                pending.append(f'{rank}:mamba_states')
            byte_count = p * states[rank]['kv_bytes_per_token'] + copies * states[rank].get('mamba_state_bytes', 0)
            if delta['load_bytes'] > byte_count and ('mamba_states' not in delta or copies == 2):
                raise RuntimeError('H2D 字节数超过 KV 与线性状态的总量')
            if delta['load_bytes'] != byte_count or delta['load_ops'] <= 0 or delta['load_seconds'] <= 0:
                pending.append(f'{rank}:load')
        elif any(delta[k] for k in ('load_bytes', 'load_seconds', 'load_ops', 'kv_tokens', 'mamba_states') if k in delta):
            raise RuntimeError('Cold/L1 不应有缓存 H2D 恢复')
    if pending:
        raise PendingMetrics('等待请求指标：' + ', '.join(pending))
    return deltas


class RankMetrics:
    """将等待、校验、归档集中在这里，Driver 只标明请求边界。"""

    def __init__(self, client, info, hybrid, path, *, gpu_timing=True):
        self.client, self.path = client, path
        self.tp, self.ep = info['tp_size'], info.get('ep_size', 1)
        self.hybrid = hybrid
        self.gpu_timing = gpu_timing
        self.chunk = info.get('chunked_prefill_size')

    def snapshot(self, stage, before=None, **expected):
        # HTTP 返回不代表所有 CUDA Event 已发布。先检查预期计数，再要求所选
        # 指标稳定 400 ms；这是保守等待策略，不是服务端的“统计提交完成”ACK。
        # 此等待发生在 generate 计时之外；50 ms 是轮询间隔，不是计时精度。
        deadline = time.monotonic() + self.client.timeout
        previous, since, reason = None, time.monotonic(), '尚无指标'
        while True:
            try:
                raw, _ = self.client.request('/metrics', stage=stage, parse_json=False)
                snapshot = parse_metrics(raw, self.tp, self.ep, self.hybrid, gpu_timing=self.gpu_timing)
                if before is not None:
                    metric_delta(before, snapshot, chunk=self.chunk, gpu_timing=self.gpu_timing, **expected)
                now = time.monotonic()
                if previous != snapshot:
                    previous, since = snapshot, now
                if now - since >= .4:
                    return snapshot
            except PendingMetrics as exc:
                previous, since, reason = None, time.monotonic(), str(exc)
            if time.monotonic() >= deadline:
                raise TimeoutError(reason)
            time.sleep(.05)

    def finish(self, before, stage, rid, **expected):
        after = self.snapshot(stage + ':metrics_after', before, **expected)
        expectation = dict(expected, chunk=self.chunk, gpu_timing=self.gpu_timing)
        delta = metric_delta(before, after, **expectation)
        record = dict(stage=stage, request_id=rid, tp=self.tp, ep=self.ep,
                      expected=expectation, before=before, after=after, deltas=delta)
        with self.path.open('a') as file:
            file.write(json.dumps(record) + '\n')
        # 时间取最慢 rank；逻辑 token 不按卡累加，物理字节数才求和。
        result = dict(kv_load_ms=max(d['load_seconds'] for d in delta.values()) * 1000,
                    kv_load_tokens=delta[0]['kv_tokens'], kv_load_bytes=sum(d['load_bytes'] for d in delta.values()),
                    kv_load_ops=max(d['load_ops'] for d in delta.values()),
                    mamba_load_states=delta[0].get('mamba_states', 0))
        if self.gpu_timing:
            result.update(prefill_forward_gpu_ms=max(d['seconds'] for d in delta.values()) * 1000,
                          prefill_forward_passes=delta[0]['passes'])
        return result
