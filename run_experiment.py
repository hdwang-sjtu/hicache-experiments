#!/usr/bin/env python3
"""独立 HTTP 实验客户端：构造输入、控制缓存、验证命中并分析结果。

服务可由 Docker、serve.sh 或其他方式启动；本脚本不管理服务生命周期。
模型权重和 SGLang 只需装在服务端，客户端仅需匹配的 tokenizer 文件。

阅读顺序：main → run → run_samples → execute_sample → validate。
  run        运行实验：准备输入 → 等待服务 → 采样 → 保存 CSV 和汇总。
  analyze    读取已有结果重新汇总，不访问模型或服务器。
  self-test  用假 tokenizer 和假服务检查脚本逻辑，不做真实模型推理。

约定：P 是可复用的历史前缀长度，S 是本次新增输入长度，S₀ 是预热用后缀。
Cold 不预热；L1 先发送 P+S₀，再发送 P+S，且 S₀ 与 S 从首个 token 分叉。
L2 先确认 CPU 备份完成，再通过实验接口逐出 GPU。
--gpu-timing 读取原生 CUDA Event 区间；--profile 单独诊断 Kernel 执行与加载重叠。
"""

import argparse
from bisect import bisect_right
from bisect import bisect_left
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


# 固定性能测试语料。保留英文原文，避免修改注释时改变既有输入及其哈希。
TEXT = """The research assistant reads a project notebook and continues a task.
The notebook describes observations, candidate explanations, and measurements.
Each experiment uses the same inputs and records its configuration and results.
The assistant compares the evidence, checks intermediate steps, and writes down
what is known and what still needs verification before choosing the next action.
"""
# CSV 列名 → SGLang meta_info 字段。服务端时间单位是秒，写 CSV 时转为毫秒。
# None 表示该项由客户端自己测量，不从服务响应提取。
TIMES = {
    "client_latency_ms": None,
    "server_e2e_ms": "e2e_latency",
    "server_first_token_ms": "first_token_latency",
    "queue_ms": "queue_time",
}
# 来自请求前后 /metrics 的差值，不是从 meta_info 里读取。
LOAD_FIELDS = ("kv_load_ms", "kv_load_tokens", "kv_load_bytes", "kv_load_ops")
FORWARD_FIELD = "prefill_forward_gpu_ms"
ENV_FIELDS = (
    "cache_valid", "environment_valid", "environment_reason",
    "measurement_start_monotonic_ns", "measurement_end_monotonic_ns",
    "measurement_start_utc_ns", "measurement_end_utc_ns",
    "gpu_monitor_max_gap_ms", "host_monitor_max_gap_ms", "unexpected_gpu_pids",
)
# CSV 字段保持英文，方便脚本处理；含义见 docs/usage.md 的“指标含义”。
FIELDS = [
    "case", "group", "attempt", "request_id", "model", "dtype", "kv_dtype",
    "prefix_length", "suffix_length", "output_length", "input_sha256",
    "prompt_tokens", "completion_tokens", "cached_tokens", "device_tokens",
    "host_tokens", "storage_tokens", "num_retractions", *TIMES,
    "forward_entry_time", "prefill_finished_time", *LOAD_FIELDS, FORWARD_FIELD,
    "device_tokens_before", "host_backed_tokens_before",
    "device_used_tokens_before", "host_used_tokens_before",
    "device_used_tokens_after", "host_used_tokens_after", "output_ids", "profiled", *ENV_FIELDS, "valid", "invalid_reason",
]


def write_json(path, value):
    """保存缩进后的 JSON，便于直接打开检查参数和原始输入。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class EnvironmentMonitor:
    """只读宿主机采样，缓存正确性与环境覆盖分别记录，不按延迟大小过滤。"""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.owner = json.loads((self.directory / "owner.json").read_text())
        self.offsets = {kind: 0 for kind in ("gpu", "host")}
        self.records = {kind: [] for kind in self.offsets}

    def refresh(self, keep_since=None):
        cutoff = time.monotonic_ns() - 120_000_000_000
        if keep_since is not None:
            cutoff = min(cutoff, keep_since)
        for kind in self.records:
            path = self.directory / f"{kind}.jsonl"
            if not path.exists():
                continue
            with path.open("rb") as file:
                file.seek(self.offsets[kind])
                while True:
                    line = file.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    self.records[kind].append(json.loads(line))
                    self.offsets[kind] = file.tell()
            self.records[kind] = [r for r in self.records[kind] if r["end_monotonic_ns"] >= cutoff]

    def unexpected_pids(self, record):
        gpu = record.get("gpus", {}).get(str(self.owner["gpu_index"]))
        if gpu is None or "compute_processes" not in gpu:
            return None
        return sorted({p["pid"] for p in gpu["compute_processes"]}
                      - set(self.owner["allowed_gpu_pids"]))

    def ready_reason(self):
        self.refresh()
        now = time.monotonic_ns()
        for kind, records in self.records.items():
            if not records:
                return f"{kind} 采样未就绪"
            latest = records[-1]
            if not latest["ok"] or now - latest["end_monotonic_ns"] > 2_000_000_000:
                return f"{kind} 采样失败或超过 2 秒未更新"
            if latest["duration_ms"] > 3000:
                return f"{kind} 查询持续超过 3 秒"
        pids = self.unexpected_pids(self.records["gpu"][-1])
        if pids is None or pids:
            return f"选定 GPU 不可见或出现额外计算进程：{pids}"
        return ""

    def wait_ready(self):
        deadline = time.monotonic() + 300
        announced = False
        while True:
            reason = self.ready_reason()
            if not reason:
                return
            if not announced:
                print("暂停请求，等待环境恢复：" + reason, flush=True)
                announced = True
            if time.monotonic() >= deadline:
                raise RuntimeError("环境持续异常超过 5 分钟：" + reason)
            time.sleep(.25)

    def evaluate(self, window):
        start, end = window["measurement_start_monotonic_ns"], window["measurement_end_monotonic_ns"]
        deadline = time.monotonic() + 3
        while True:
            self.refresh(start - 3_000_000_000)
            if all(rs and rs[-1]["end_monotonic_ns"] >= end for rs in self.records.values()):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(.1)
        reasons, extra, unexpected = [], {}, set()
        for kind, records in self.records.items():
            times = [r["end_monotonic_ns"] for r in records]
            left, right = bisect_right(times, start)-1, bisect_left(times, end)
            extra[f"{kind}_monitor_max_gap_ms"] = None
            if left < 0 or right >= len(times):
                reasons.append(f"{kind} 缺少请求前后采样覆盖")
                continue
            selected = records[left:right+1]
            gap = max((b-a for a,b in zip(times[left:right+1],times[left+1:right+1])), default=0)
            extra[f"{kind}_monitor_max_gap_ms"] = gap / 1e6
            if start-times[left] > 2_000_000_000 or times[right]-end > 2_000_000_000 or gap > 3_000_000_000:
                reasons.append(f"{kind} 覆盖边界超过 2 秒或采样间隔超过 3 秒")
            if any(not r["ok"] or r["duration_ms"] > 3000 for r in selected):
                reasons.append(f"{kind} 采样失败或查询持续超过 3 秒")
            if kind == "gpu":
                for record in selected:
                    pids = self.unexpected_pids(record)
                    if pids is None:
                        reasons.append("选定 GPU 缺少进程信息")
                    else:
                        unexpected.update(pids)
        if unexpected:
            reasons.append(f"选定 GPU 出现额外计算进程：{sorted(unexpected)}")
        return dict(environment_valid=not reasons,environment_reason="; ".join(dict.fromkeys(reasons)),
                    unexpected_gpu_pids=json.dumps(sorted(unexpected)),**extra)


def sample_gpu_devices(nvml, selected_index):
    """逐卡记录错误；其他卡故障不能丢掉实验卡的有效采样。"""
    gpus = {}
    for index in range(nvml.nvmlDeviceGetCount()):
        try:
            # 每轮重新取句柄，初始化时已有故障卡也不影响其他卡。
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            gpu = dict(ok=True, uuid=nvml.nvmlDeviceGetUUID(handle), utilization_gpu=util.gpu,
                       utilization_memory=util.memory, memory_used_bytes=memory.used,
                       memory_total_bytes=memory.total,
                       compute_processes=[dict(pid=p.pid, used_bytes=p.usedGpuMemory) for p in processes])
            # 可选传感器不支持时保留错误，不把缺失值写成假零。
            gpu["optional_errors"] = {}
            for key, name, params in (
                ("temperature_c", "nvmlDeviceGetTemperature", (nvml.NVML_TEMPERATURE_GPU,)),
                ("power_mw", "nvmlDeviceGetPowerUsage", ()),
                ("sm_clock_mhz", "nvmlDeviceGetClockInfo", (nvml.NVML_CLOCK_SM,)),
                ("memory_clock_mhz", "nvmlDeviceGetClockInfo", (nvml.NVML_CLOCK_MEM,)),
                ("throttle_reasons", "nvmlDeviceGetCurrentClocksThrottleReasons", ()),
            ):
                try:
                    gpu[key] = getattr(nvml, name)(handle, *params)
                except (AttributeError, nvml.NVMLError) as error:
                    gpu[key] = None
                    gpu["optional_errors"][key] = str(error)
            gpus[str(index)] = gpu
        except nvml.NVMLError as error:
            gpus[str(index)] = dict(ok=False, error=repr(error))
    selected = gpus.get(str(selected_index), {})
    result = dict(ok=selected.get("ok", False), gpus=gpus)
    if not result["ok"]:
        result["error"] = selected.get("error", "选定 GPU 不可见")
    return result


def monitor(args):
    """GPU 与主机分别运行本函数，NVML 阻塞不会停止主机心跳。"""
    if args.interval <= 0:
        raise ValueError("采样间隔必须大于零")
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    owner = json.loads((directory / "owner.json").read_text())
    if args.kind == "gpu":
        import pynvml as nvml
        nvml.nvmlInit()
    else:
        import psutil
        psutil.cpu_percent(percpu=True)
    with (directory / f"{args.kind}.jsonl").open("a", buffering=1) as log:
        while not (directory / "stop").exists():
            start = time.monotonic_ns()
            record = dict(start_monotonic_ns=start,start_utc_ns=time.time_ns(),ok=True)
            try:
                if args.kind == "gpu":
                    record.update(sample_gpu_devices(nvml, owner["gpu_index"]))
                else:
                    record.update(cpu_percent_per_core=psutil.cpu_percent(percpu=True),
                                  load_average=os.getloadavg(),memory=psutil.virtual_memory()._asdict(),
                                  disk_io=psutil.disk_io_counters()._asdict(),pressure={})
                    for kind in ("cpu","memory","io"):
                        path = Path('/proc/pressure') / kind
                        record["pressure"][kind] = path.read_text() if path.exists() else None
                    record["scheduler"] = {}
                    try:
                        pid = owner["scheduler_pid"]
                        process = psutil.Process(pid)
                        record["scheduler"] = dict(pid=pid,cpu_times=process.cpu_times()._asdict(),
                                                   ctx_switches=process.num_ctx_switches()._asdict(),
                                                   memory=process.memory_info()._asdict(),
                                                   schedstat=(Path('/proc')/str(pid)/'schedstat').read_text())
                    except (psutil.Error,OSError) as error:
                        record["scheduler"]["error"] = str(error)
            except Exception as error:
                record.update(ok=False,error=repr(error))
            record.update(end_monotonic_ns=time.monotonic_ns(),end_utc_ns=time.time_ns())
            record["duration_ms"] = (record["end_monotonic_ns"]-start)/1e6
            line = json.dumps(record,ensure_ascii=False)
            log.write(line+'\n')
            temp = directory / f'{args.kind}.latest.tmp'
            temp.write_text(line+'\n')
            temp.replace(directory / f'{args.kind}.latest.json')
            time.sleep(max(0,args.interval-(time.monotonic_ns()-start)/1e9))


def digest(value):
    """计算稳定的内容哈希，用来确认 Cold 和 L1 测量输入完全一致。"""
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def positive_length(value):
    """解析命令行中的正整数或 k 后缀；例如 8k 等于 8192。"""
    try:
        number = int(value[:-1]) * 1024 if value.lower().endswith("k") else int(value)
        if number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Expected a positive integer or e.g. 8k (8192)")


def tile(tokens, length):
    """重复一段已分词的序列，再截取到精确长度；不重新拼接文本分词。"""
    if not tokens:
        raise ValueError("Tokenized text is empty")
    return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]


def make_cases(tokenizer, prefixes, suffixes, text):
    """为每个 P/S 组合保存 prefix、测量 suffix、预热 suffix 和输入哈希。"""
    unit = tokenizer.encode(text, add_special_tokens=False)
    suffix = tokenizer.encode(
        "The next task is to review the recorded observations. ",
        add_special_tokens=False,
    )
    warm = tokenizer.encode(
        "Another task is to summarize the previous measurements. ",
        add_special_tokens=False,
    )
    special = set(tokenizer.all_special_ids)
    if not unit or not suffix or not warm or special.intersection(unit + suffix + warm):
        raise ValueError("Input text must tokenize to nonempty, ordinary tokens")
    # 两句文本分词后仍可能以相同 token 开头，因此显式选择不同的首 token。
    # 这样 P+S₀ 和 P+S 只共享 P；不会把测量 suffix 也算成缓存命中。
    alternate = next((token for token in warm if token != suffix[0]), None)
    if alternate is None:
        raise ValueError("Cannot construct suffixes with different first tokens")
    cases = []
    for p in prefixes:
        for s in suffixes:
            measured, prepared = tile(suffix, s), tile(warm, s)
            prepared[0] = alternate
            prefix = tile(unit, p)
            cases.append(
                dict(
                    id=f"p{p}_s{s}",
                    prefix=prefix,
                    suffix=measured,
                    warm_suffix=prepared,
                    input_sha256=digest(prefix + measured),
                )
            )
    return cases


class Client:
    """串行 HTTP 客户端，同时把控制操作和原始响应写入 responses.jsonl。"""

    def __init__(self, base_url, timeout, events):
        self.base_url, self.timeout, self.events = base_url.rstrip("/"), timeout, events
        self.hicache = False  # run() 根据实际服务配置设置，不凭命令行猜测。
        self.profile_dir = None
        self.gpu_timing = False
        self.environment = None
        self.last_request_window = None

    def request(self, path, payload=None, *, stage="control", parse_json=True):
        """发送一次请求，返回响应及客户端耗时；stage 用于区分记录的用途。

        计时覆盖 HTTP 请求到读完响应，不包括此前的输入构造和 JSON 编码。
        它包含网络、排队和推理耗时，不能当成纯 GPU Prefill 时间。
        """
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        mono_started, utc_started = time.monotonic_ns(), time.time_ns()
        started = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            # 请求失败后缓存状态可能不确定：记录错误并终止，由调用方保存失败状态。
            body = (
                exc.read().decode(errors="replace")
                if isinstance(exc, urllib.error.HTTPError)
                else ""
            )
            self.events.write(
                json.dumps(
                    dict(
                        stage=stage,
                        path=path,
                        request_id=(payload or {}).get("rid"),
                        error=str(exc),
                        error_body=body,
                    )
                ) + "\n"
            )
            self.events.flush()
            hint = ""
            if isinstance(exc, urllib.error.HTTPError) and path == "/debug/hicache" and exc.code == 403:
                hint = "；缓存控制仅接受服务端本机连接，远程实验请通过 SSH 隧道访问。"
            raise RuntimeError(f"{stage} {path}: {exc} {body}{hint}") from exc
        elapsed = (time.perf_counter_ns() - started) / 1e6
        self.last_request_window = dict(
            measurement_start_monotonic_ns=mono_started,
            measurement_end_monotonic_ns=time.monotonic_ns(),
            measurement_start_utc_ns=utc_started,
            measurement_end_utc_ns=time.time_ns(),
        )
        result = json.loads(raw) if parse_json else raw.decode()
        self.events.write(
            json.dumps(
                dict(
                    stage=stage,
                    path=path,
                    request_id=(payload or {}).get("rid"),
                    client_latency_ms=elapsed,
                    **self.last_request_window,
                    response=result,
                ),
                ensure_ascii=False,
            ) + "\n"
        )
        self.events.flush()
        return result, elapsed

    def flush(self):
        """清空本阶段的缓存。只在样本准备时调用，不计入测量请求延迟。"""
        # urlopen 会拒绝非成功 HTTP 状态；清空失败时不继续测量。
        # 注意：该接口会同时重置多级缓存，后续不能用它构造 L2 Hit。
        self.request("/flush_cache?timeout=30", {}, parse_json=False)

    def wait_cache(self, prefix_ids, *, demote=False):
        """在测量区间外等待稳定状态；demote 忙时无副作用，可安全重试。"""
        deadline = time.monotonic() + self.timeout
        while True:
            result, _ = self.request(
                "/debug/hicache",
                dict(action="demote" if demote else "inspect", prefix_ids=prefix_ids),
                stage="demote" if demote else "cache_state",
            )
            if not result.get("success"):
                raise RuntimeError("缓存控制失败：" + result.get("message", str(result)))
            if result.get("ready"):
                return result["state"]
            if time.monotonic() >= deadline:
                raise TimeoutError("等待缓存备份/恢复结束超时")
            time.sleep(.02)

    def load_metrics(self, stage):
        text, _ = self.request("/metrics", stage=stage, parse_json=False)
        result = parse_load_metrics(text)
        if self.gpu_timing:
            result[FORWARD_FIELD] = parse_forward_metric(text)
        return result

    def generate(self, ids, output_length, stage):
        """直接发送 token IDs，避免聊天模板或重新分词改变精确输入长度。"""
        request_id = "cache-exp-" + uuid.uuid4().hex
        response, elapsed = self.request(
            "/generate",
            dict(
                rid=request_id,
                input_ids=ids,
                stream=False,
                # 固定输出长度；不因模型提前生成 EOS 而缩短本次请求。
                sampling_params=dict(
                    temperature=0,
                    max_new_tokens=output_length,
                    ignore_eos=True,
                ),
            ),
            stage=stage,
        )
        return response, elapsed, request_id


def parse_load_metrics(text):
    """解析单 rank 的四个既有 H2D 指标；缺失、重复或非有限数值不能当作零。"""
    names = {
        "sglang:load_back_duration_seconds_sum": "kv_load_ms",
        "sglang:load_back_tokens_total": "kv_load_tokens",
        "sglang:load_back_bytes_total": "kv_load_bytes",
        "sglang:load_back_duration_seconds_count": "kv_load_ops",
    }
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([\w:]+)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?', line)
        if match is None or match[1] not in names:
            continue
        labels = dict(re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[2] or ""))
        if match[1].endswith("tokens_total") and labels.get("pool") != "kv":
            continue
        if any(labels.get(rank) != "0" for rank in ("tp_rank", "pp_rank", "dp_rank")):
            raise RuntimeError("H2D 指标包含非零 rank，不能归属到单卡实验")
        key = names[match[1]]
        value = float(match[3])
        if key in result or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"H2D 指标重复或数值无效：{key}")
        if key != "kv_load_ms" and not value.is_integer():
            raise RuntimeError(f"H2D 计数不是整数：{key}")
        result[key] = value * 1000 if key == "kv_load_ms" else int(value)
    missing = set(LOAD_FIELDS) - result.keys()
    if missing:
        raise RuntimeError(f"缺少 H2D 指标：{sorted(missing)}；检查实验接口和 --enable-metrics")
    return result


def parse_forward_metric(text):
    """读取原生 extend CUDA Event 累计区间；缺失表示计时未启用，不能填零。"""
    values = []
    for line in text.splitlines():
        match = re.fullmatch(r'sglang:forward_execution_seconds_total\{(.*)\}\s+(\S+)', line)
        if match is None:
            continue
        labels = dict(re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[1]))
        if labels.get("category") != "extend":
            continue
        # 原生 forward 指标没有 dp_rank；DP=1 已由 check_server 强制检查。
        if any(labels.get(rank) != "0" for rank in ("tp_rank", "pp_rank")) or any(
            rank in labels and labels[rank] != "0" for rank in ("dp_rank", "moe_ep_rank")
        ):
            raise RuntimeError("Forward 指标包含非零 rank")
        values.append(float(match[2]) * 1000)
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
        raise RuntimeError("缺少有效的 extend GPU 时间；以 FORWARD_TIMING=1 重启服务")
    return values[0]


def load_deltas(before, after, p, bytes_per_token, group):
    """将累计指标转为一次请求的值，并验证传输量；不把提交时间当完成时间。"""
    values = {key: after[key] - before[key] for key in LOAD_FIELDS}
    errors = []
    if any(value < 0 for value in values.values()):
        errors.append("H2D 累计指标倒退或被重置")
    expected = p if group == "l2" else 0
    if values["kv_load_tokens"] != expected:
        errors.append(f"H2D tokens: expected {expected}, got {values['kv_load_tokens']}")
    if values["kv_load_bytes"] != expected * bytes_per_token:
        errors.append("H2D 字节数与实际池的每 token 字节数不符")
    if group == "l2":
        if values["kv_load_ops"] <= 0 or values["kv_load_ms"] <= 0:
            errors.append("L2 没有已完成的 H2D CUDA Event 计时")
    elif values["kv_load_ops"] != 0 or values["kv_load_ms"] != 0:
        errors.append("Cold/L1 不应出现 H2D 加载")
    return values, errors


def validate(response, p, s, output_length, group):
    """判断响应能否计入指定组，返回无效原因列表和各层命中数量。

    Cold：总命中与各层命中均为 0；L1：总命中和 device 命中恰好为 P。
    长度、回撤次数和已有时间指标也必须符合要求；缺失数据不当作零。
    """
    meta = response.get("meta_info", {})
    errors = []
    cached = meta.get("cached_tokens")
    details = meta.get("cached_tokens_details")
    # SGLang 在完全未命中时可省略分层明细；总命中为 0 才能安全补零。
    if details is None and cached == 0:
        details = dict(device=0, host=0, storage=0)
    if not isinstance(details, dict):
        details = {}
    expected = p if group in ("l1", "l2") else 0
    checks = {
        "prompt_tokens": (meta.get("prompt_tokens"), p + s),
        "completion_tokens": (meta.get("completion_tokens"), output_length),
        "cached_tokens": (cached, expected),
        "device": (details.get("device"), p if group == "l1" else 0),
        "host": (details.get("host"), p if group == "l2" else 0),
        "storage": (details.get("storage", 0), 0),
        "num_retractions": (meta.get("num_retractions"), 0),
    }
    for key, (actual, target) in checks.items():
        if type(actual) is not int or actual != target:
            errors.append(f"{key}: expected {target}, got {actual!r}")
    finish = meta.get("finish_reason") or {}
    # ignore_eos=True 时应按指定长度正常结束，而不是报错、中止或提前停止。
    if finish.get("type") != "length":
        errors.append(f"unexpected finish_reason: {finish!r}")
    for key in filter(None, TIMES.values()):
        value = meta.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            errors.append(f"missing/invalid timing: {key}")
    return errors, details


def execute_sample(client, case, group, output_length, stage="measure"):
    """执行一个完整样本：清空 → 按组准备 → 测量 P+S → 校验并组装 CSV 行。

    Cold 直接测量；L1 先用 P+S₀ 把历史 P 留在 GPU，再测量 P+S。
    清空与准备请求均不计入测量请求的延迟。每次重复都重新准备缓存。
    """
    p, s = len(case["prefix"]), len(case["suffix"])
    environment = getattr(client, "environment", None)
    if environment is not None:
        environment.wait_ready()
    client.flush()
    if group in ("l1", "l2"):
        # 预热请求本身应是 Cold；准备失败意味着测量起点不可靠，直接终止。
        prepared, _, _ = client.generate(case["prefix"] + case["warm_suffix"], 1, "prepare")
        errors, _ = validate(prepared, p, s, 1, "cold")
        if errors:
            raise RuntimeError("Preparation failed: " + "; ".join(errors))
    before_state = None
    before_metrics = None
    if getattr(client, "hicache", False):
        # CPU 副本存在不等于搬运已完成；接口同时检查 Scheduler 和传输状态。
        before_state = client.wait_cache(case["prefix"])
        expected = p if group in ("l1", "l2") else 0
        if (
            before_state["device_tokens"] != expected
            or before_state["host_backed_tokens"] != expected
            or (expected and not before_state["backup_complete"])
        ):
            raise RuntimeError(f"测量前缓存状态不符：{before_state}")
        if group == "l2":
            before_state = client.wait_cache(case["prefix"], demote=True)
            if (
                before_state["device_tokens"] != 0
                or before_state["host_hit_tokens"] != p
                or before_state["host_backed_tokens"] != p
            ):
                raise RuntimeError(f"Demote 未产生纯 L2 起点：{before_state}")
        if environment is not None:
            environment.wait_ready()
        before_metrics = client.load_metrics("metrics_before")
    # 无论哪一组，发送的测量输入都完全相同。
    # 只采集 measure，请求预热、备份和 demote 都在 Profiler 区间外。
    profiled = stage == "measure" and bool(getattr(client, "profile_dir", None))
    if profiled:
        client.request("/start_profile", dict(
            output_dir=str(client.profile_dir), activities=["CPU", "GPU"],
            with_stack=False, record_shapes=False,
            profile_id=f"{case['id']}-{group}-{uuid.uuid4().hex[:8]}",
            detailed_annotations=True,
        ), stage="profile_start", parse_json=False)
    try:
        result, elapsed, rid = client.generate(
            case["prefix"] + case["suffix"], output_length, stage
        )
        window = dict(getattr(client, "last_request_window", None) or {})
    finally:
        if profiled:
            client.request("/stop_profile", {}, stage="profile_stop", parse_json=False)
    errors, hits = validate(result, p, s, output_length, group)
    extra = {}
    if before_state is not None:
        # 请求结束后等 ACK 被处理再读终值；等待本身不计入客户端请求延迟。
        after_state = client.wait_cache(case["prefix"])
        after_metrics = client.load_metrics("metrics_after")
        extra, load_errors = load_deltas(
            before_metrics, after_metrics, p, before_state["kv_bytes_per_token"], group
        )
        errors.extend(load_errors)
        if FORWARD_FIELD in before_metrics:
            extra[FORWARD_FIELD] = after_metrics[FORWARD_FIELD] - before_metrics[FORWARD_FIELD]
            if extra[FORWARD_FIELD] <= 0:
                errors.append("Prefill CUDA Event 指标未增加或被重置")
        output_ids = result.get("output_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != output_length
            or any(type(token) is not int for token in output_ids)
        ):
            errors.append("缺少生成 token IDs，检查 --return-output-ids")
        extra.update(
            device_tokens_before=before_state["device_tokens"],
            host_backed_tokens_before=before_state["host_backed_tokens"],
            device_used_tokens_before=before_state["device_used_tokens"],
            host_used_tokens_before=before_state["host_used_tokens"],
            device_used_tokens_after=after_state["device_used_tokens"],
            host_used_tokens_after=after_state["host_used_tokens"],
        )
    meta = result.get("meta_info", {})
    cache_valid = not errors
    environment_result = {}
    if environment is not None and stage == "measure":
        environment_result = environment.evaluate(window)
        if not environment_result["environment_valid"]:
            errors.append(environment_result["environment_reason"])
    row = dict(
        case=case["id"],
        group=group,
        request_id=rid,
        prefix_length=p,
        suffix_length=s,
        output_length=output_length,
        input_sha256=case["input_sha256"],
        client_latency_ms=elapsed,
        device_tokens=hits.get("device"),
        host_tokens=hits.get("host"),
        storage_tokens=hits.get("storage", 0),
        valid=not errors,
        invalid_reason="; ".join(errors),
        output_ids=json.dumps(result.get("output_ids")),
        profiled=profiled,
        cache_valid=cache_valid,
        **window,
        **environment_result,
    )
    row.update(extra)
    # 原始时间戳按秒保存；延迟列单独从秒换算为毫秒，避免混淆单位。
    for key in (
        "prompt_tokens", "completion_tokens", "cached_tokens", "num_retractions",
        "forward_entry_time", "prefill_finished_time",
    ):
        row[key] = meta.get(key)
    for column, key in TIMES.items():
        if key:
            value = meta.get(key)
            row[column] = value * 1000 if isinstance(value, (int, float)) else None
    return row


def check_server(info, cases, output_length, groups=None):
    """开始采样前检查服务配置、上下文长度、页对齐和可见的 KV 容量。"""
    # 单卡、单请求使累计指标差值可以归属到当前测量请求。
    required = dict(
        enable_metrics=True,
        disable_radix_cache=False,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        max_running_requests=1,
    )
    for key, expected in required.items():
        if info.get(key) != expected:
            raise ValueError(f"Experiment requires server {key}={expected!r}; got {info.get(key)!r}")
    if info.get("enable_hierarchical_cache"):
        if (
            not info.get("enable_hicache_experiment")
            or info.get("hicache_host_memory_mode") != "cache"
            or info.get("hicache_write_policy") != "write_through"
            or info.get("hicache_storage_backend") is not None
        ):
            raise ValueError("HiCache 需启用实验接口、cache/write_through 模式且关闭 L3")
    elif "l2" in (groups or []):
        raise ValueError("L2 实验需要使用 HICACHE=1 重启服务")
    if info.get("speculative_algorithm"):
        raise ValueError("Disable speculative decoding for this experiment")
    if info.get("enable_session_radix_cache") or info.get("enable_streaming_session"):
        raise ValueError("本实验使用普通请求，请关闭会话缓存和 streaming session")
    context = info.get("context_length")
    page = info.get("page_size")
    if not isinstance(context, int) or not isinstance(page, int) or page < 1:
        raise ValueError("Server must report context_length and page_size")
    for case in cases:
        p, s = len(case["prefix"]), len(case["suffix"])
        # Prefix 需按物理页对齐，才能要求精确命中 P；输入和输出都需预留容量。
        if p % page or p + s + output_length > context:
            raise ValueError(f"{case['id']}: prefix must be page aligned; total length must fit context")
        capacity = info.get("max_total_num_tokens", info.get("max_total_tokens"))
        if isinstance(capacity, int) and p + s + output_length > capacity:
            raise ValueError(f"{case['id']}: input and output exceed KV capacity {capacity}")


def run_samples(client, cases, args, writer, file, common):
    """按 P/S 组合采样，直到每组达到指定有效次数，或超过最大尝试次数。"""
    rng = random.Random(args.seed)
    for case in cases:
        reference_output = None
        # 这里的 warmup 用于预热运行路径，不写入 samples.csv。
        # 它与 execute_sample 内每次 L1 都要执行的 prepare 请求不同。
        for _ in range(args.warmup_rounds):
            for group in args.groups:
                row = execute_sample(client, case, group, args.output_length, "warmup")
                if not row["valid"]:
                    raise RuntimeError("Warmup failed: " + row["invalid_reason"])
                if group == "cold" or reference_output is None:
                    # 有 Cold 对照时以其输出为准，不让随机先运行的 L2 成为错误参考。
                    reference_output = row["output_ids"]
        valid = dict.fromkeys(args.groups, 0)
        attempts = dict.fromkeys(args.groups, 0)
        while any(valid[g] < args.repeats for g in args.groups):
            # 每轮只运行尚未收满的组，并打乱顺序，减少固定先后顺序的影响。
            groups = [g for g in args.groups if valid[g] < args.repeats]
            rng.shuffle(groups)
            for group in groups:
                if attempts[group] >= args.max_attempts:
                    raise RuntimeError(
                        f"{case['id']}/{group}: insufficient valid samples "
                        f"after {args.max_attempts} attempts"
                    )
                attempts[group] += 1
                row = execute_sample(client, case, group, args.output_length)
                if row["valid"]:
                    if reference_output is None:
                        reference_output = row["output_ids"]
                    elif row["output_ids"] != reference_output:
                        row["valid"] = False
                        row["cache_valid"] = False
                        row["invalid_reason"] = "相同测量输入的生成 token 与本组实验参考结果不一致"
                row.update(common, attempt=attempts[group])
                # 无效样本也立即落盘，但只增加 attempts，不增加 valid。
                writer.writerow(row)
                file.flush()
                valid[group] += int(row["valid"])
                print(
                    f"{case['id']} {group}: valid {valid[group]}/{args.repeats}, "
                    f"client={row['client_latency_ms']:.2f} ms {row['invalid_reason']}",
                    flush=True,
                )


def percentile(values, fraction):
    """在排序后的相邻样本间线性插值；fraction=0.95 表示 P95。"""
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summarize_rows(rows, config, metrics, valid_column="valid"):
    """同一原始表按正式有效或仅缓存有效分别统计，不删除环境异常尝试。"""
    summaries = []
    for p in config["arguments"]["prefix_lengths"]:
        for s in config["arguments"]["suffix_lengths"]:
            for group in config["arguments"]["groups"]:
                selected = [
                    r for r in rows
                    if int(r["prefix_length"]) == p
                    and int(r["suffix_length"]) == s
                    and r["group"] == group
                ]
                # CSV 读出的布尔值是字符串；只用有效样本计算延迟分布。
                valid = [r for r in selected if r.get(valid_column, r["valid"]) == "True"]
                for metric in metrics:
                    values = [float(r[metric]) for r in valid if r[metric]]
                    summaries.append(
                        dict(
                            prefix_length=p,
                            suffix_length=s,
                            group=group,
                            metric=metric,
                            attempted=len(selected),
                            valid=len(valid),
                            invalid=len(selected) - len(valid),
                            required=config["arguments"]["repeats"],
                            complete=len(valid) >= config["arguments"]["repeats"],
                            median=statistics.median(values) if values else None,
                            p95=percentile(values, .95) if values else None,
                        )
                    )
    return summaries


def analyze(directory, no_plot=False):
    """从已保存的 CSV 重新计算汇总和图表，不重新运行任何模型请求。"""
    config = json.loads((directory / "config.json").read_text())
    with (directory / "samples.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    # 老结果没有 H2D 列，不能把“未测量”填成零。
    metrics = list(TIMES) + [key for key in (*LOAD_FIELDS, FORWARD_FIELD) if any(r.get(key) for r in rows)]
    summaries = summarize_rows(rows, config, metrics)
    outputs = [("summary.csv", summaries)]
    if any(r.get("cache_valid") for r in rows):
        outputs.append(("all_cache_valid_summary.csv", summarize_rows(rows, config, metrics, "cache_valid")))
    for name, data in outputs:
        with (directory / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    if not no_plot:
        # Docker 中按 UID 运行的用户可能没有可写 home，缓存改放结果目录。
        os.environ.setdefault("MPLCONFIGDIR", str(directory.resolve() / ".matplotlib"))
        try:
            import matplotlib

            # 容器通常没有桌面环境，直接保存图片，不弹出图形窗口。
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("CSV analysis complete; plot unavailable: install matplotlib and rerun analyze.", file=sys.stderr)
            return
        suffixes = config["arguments"]["suffix_lengths"]
        prefixes = config["arguments"]["prefix_lengths"]
        scan_suffix = len(prefixes) == 1 and len(suffixes) > 1
        xkey = "suffix_length" if scan_suffix else "prefix_length"
        fixed_key = "prefix_length" if scan_suffix else "suffix_length"
        panels = prefixes if scan_suffix else suffixes
        # 固定 P 时横轴改为 S；否则每个 S 一张 P 扫描子图。
        # 图中文字保留英文，避免依赖容器中的中文字库。
        columns = min(3, len(panels))
        nrows = math.ceil(len(panels) / columns)
        fig, axes = plt.subplots(nrows, columns, figsize=(6 * columns, 4 * nrows), squeeze=False)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        for ax, fixed in zip(axes.flat, panels):
            for group in config["arguments"]["groups"]:
                data = sorted(
                    [
                        r for r in summaries
                        if r[fixed_key] == fixed
                        and r["group"] == group
                        and r["metric"] == "server_first_token_ms"
                        and r["complete"]
                    ],
                    key=lambda r: r[xkey],
                )
                if data:
                    ax.plot(
                        [r[xkey] for r in data],
                        [r["median"] for r in data],
                        "o-",
                        label=f"{group} median",
                    )
                    ax.plot(
                        [r[xkey] for r in data],
                        [r["p95"] for r in data],
                        "x--",
                        label=f"{group} p95",
                    )
            ax.set(
                title=f"{'Prefix' if scan_suffix else 'Suffix'} = {fixed} tokens",
                xlabel="Suffix tokens" if scan_suffix else "Prefix tokens",
                ylabel="Server first-token latency (ms)",
            )
            ax.grid(alpha=.3)
            ax.set_yscale("log")
            if ax.lines:
                ax.legend()
        status = config.get("status", "unknown")
        if config["arguments"].get("profile"):
            status += "; PROFILER ENABLED (diagnostic only)"
        note = config["arguments"].get("note", "")
        title = f"{config.get('model', 'model')} — run: {status}; complete groups only"
        fig.suptitle(title + ("\n" + note if note else ""))
        fig.tight_layout()
        fig.savefig(directory / "latency.png", dpi=160)
        plt.close(fig)
        if FORWARD_FIELD in metrics:
            plot_timing(directory, config, summaries, plt)
        if len(prefixes) > 1 and len(suffixes) > 1:
            plot_heatmaps(directory, config, summaries, plt)


def plot_timing(directory, config, summaries, plt):
    """并排画流区间，避免把可重叠的 H2D 和 forward 画成串行堆叠图。"""
    prefixes = config["arguments"]["prefix_lengths"]
    suffixes = config["arguments"]["suffix_lengths"]
    scan_prefix = len(prefixes) > 1
    xkey = "prefix_length" if scan_prefix else "suffix_length"
    fixed_key = "suffix_length" if scan_prefix else "prefix_length"
    panels = suffixes if scan_prefix else prefixes
    columns = min(3, len(panels))
    nrows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(7 * columns, 4 * nrows), squeeze=False)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for ax, fixed in zip(axes.flat, panels):
        for group, metric, label in (
            ("l1", FORWARD_FIELD, "L1 forward span"),
            ("l2", "kv_load_ms", "L2 H2D span"),
            ("l2", FORWARD_FIELD, "L2 forward span (includes waits)"),
        ):
            data = sorted((r for r in summaries if r[fixed_key] == fixed
                           and r["group"] == group and r["metric"] == metric
                           and r["complete"] and r["median"] is not None),
                          key=lambda r: r[xkey])
            if data:
                medians = [r["median"] for r in data]
                ax.errorbar([r[xkey] for r in data], medians,
                            yerr=[[0] * len(data), [max(0, r["p95"] - r["median"]) for r in data]],
                            marker="o", capsize=3, label=label)
        ax.set(title=f"{'Suffix' if scan_prefix else 'Prefix'} = {fixed} tokens",
               xlabel="Prefix tokens" if scan_prefix else "Suffix tokens",
               ylabel="CUDA Event interval (ms)")
        ax.grid(alpha=.3)
        if ax.lines:
            ax.legend()
    note = config["arguments"].get("note", "")
    title = "Median / P95 stream intervals; H2D and forward may overlap"
    fig.suptitle(title + ("\n" + note if note else ""))
    fig.tight_layout()
    fig.savefig(directory / "timing.png", dpi=160)
    plt.close(fig)


def plot_heatmaps(directory, config, summaries, plt):
    """完整 P/S 矩阵按类别显示；缺测格留空，不以插值伪造测量。"""
    from matplotlib.colors import LogNorm
    prefixes = config["arguments"]["prefix_lengths"]
    suffixes = config["arguments"]["suffix_lengths"]
    for metric in (FORWARD_FIELD, "server_first_token_ms"):
        for statistic in ("median", "p95"):
            values = {(r["prefix_length"], r["suffix_length"]): r[statistic] for r in summaries
                      if r["group"] == "l1" and r["metric"] == metric and r["complete"]
                      and r[statistic] is not None}
            finite = [v for v in values.values() if v > 0]
            if not finite:
                continue
            grid = [[values.get((p, s), float("nan")) for s in suffixes] for p in prefixes]
            fig, ax = plt.subplots(figsize=(12, max(5, len(prefixes) * .45)))
            chart = ax.imshow(grid, origin="lower", aspect="auto",
                              norm=LogNorm(vmin=min(finite), vmax=max(max(finite), min(finite)*1.001)))
            ax.set(xticks=list(range(len(suffixes))), xticklabels=suffixes,
                   yticks=list(range(len(prefixes))), yticklabels=[f"{p//1024}K" for p in prefixes],
                   xlabel="Suffix tokens", ylabel="Prefix tokens")
            for i, p in enumerate(prefixes):
                for j, s in enumerate(suffixes):
                    value = values.get((p, s))
                    if value is not None:
                        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7,
                                color="white" if chart.norm(value) < .6 else "black")
            fig.colorbar(chart, ax=ax, label="ms (log color scale)")
            fig.suptitle(f"L1 {metric} — {statistic}\n" + config["arguments"].get("note", ""))
            fig.tight_layout()
            fig.savefig(directory / f"{metric}_{statistic}_heatmap.png", dpi=160)
            plt.close(fig)


def merge_intervals(intervals):
    """同一时钟上的区间先取并集，避免并发 Kernel 被重复累计。"""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def overlap_duration(left, right):
    """输入为已合并、按起点排序的区间，单位沿用 trace 的微秒。"""
    i = j = 0
    total = 0.
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def trace_intervals(events):
    """通过 CUDA correlation 和 CPU step 范围归属 forward Kernel，不硬编码 stream ID。"""
    steps = {}
    for event in events:
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith("step[EXTEND "):
            steps.setdefault((event["pid"], event["tid"]), []).append(event)
    for group in steps.values():
        group.sort(key=lambda e: e["ts"])
    starts = {key: [e["ts"] for e in group] for key, group in steps.items()}
    forward_correlations = set()
    for event in events:
        key = (event.get("pid"), event.get("tid"))
        if event.get("cat") not in ("cuda_runtime", "cuda_driver") or key not in steps:
            continue
        index = bisect_right(starts[key], event["ts"]) - 1
        if index >= 0 and event["ts"] <= steps[key][index]["ts"] + steps[key][index]["dur"]:
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                forward_correlations.add(correlation)
    forward, h2d, h2d_names = [], [], set()
    forward_count = h2d_count = 0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        if event.get("args", {}).get("device") != 0:
            raise ValueError("只支持单卡 trace 的区间归属")
        interval = (event["ts"], event["ts"] + event["dur"])
        # 对应 pool_host/mha.py 的 kernel 后端逐层恢复；不把 D2H relayout 算成 H2D。
        if re.search(r"hicache_transfer_per_layer|transfer_kv_per_layer", event["name"]):
            h2d.append(interval)
            h2d_names.add(event["name"])
            h2d_count += 1
        elif event.get("args", {}).get("correlation") in forward_correlations:
            forward.append(interval)
            forward_count += 1
    tokens = sum(int(re.search(r"\btoks=(\d+)", e["name"])[1])
                 for group in steps.values() for e in group)
    return merge_intervals(forward), merge_intervals(h2d), dict(
        forward_kernel_count=forward_count, h2d_kernel_count=h2d_count,
        forward_input_tokens=tokens, h2d_kernel_names=sorted(h2d_names),
    )


def analyze_traces(directory, no_plot=False):
    """诊断 trace：计算 Kernel 忙碌区间与 H2D 的交集，结果与正式性能 CSV 分开。"""
    config = json.loads((directory / "config.json").read_text())
    if not config["arguments"].get("profile") or config["status"] != "complete":
        raise ValueError("需要完整的 --profile 诊断运行目录")
    with (directory / "samples.csv").open() as file:
        samples = list(csv.DictReader(file))
    results, timelines = [], []
    for path in sorted((directory / "traces").glob("*.trace.json.gz")):
        match = re.match(r"p(\d+)_s(\d+)-(cold|l1|l2)-", path.name)
        if match is None:
            raise ValueError(f"无法识别 trace 输入与组别：{path.name}")
        p, s, group = int(match[1]), int(match[2]), match[3]
        matching = [r for r in samples if int(r["prefix_length"]) == p
                    and int(r["suffix_length"]) == s and r["group"] == group]
        if len(matching) != 1 or matching[0]["valid"] != "True":
            raise ValueError("每组需恰好一个有效诊断请求，避免 trace 与样本归属歧义")
        sample = matching[0]
        with gzip.open(path, "rt") as file:
            forward, h2d, details = trace_intervals(json.load(file)["traceEvents"])
        if not forward or bool(h2d) != (group == "l2"):
            raise ValueError(f"GPU trace 不完整或 H2D Kernel 未识别：{path.name}")
        if details["forward_input_tokens"] != (p + s if group == "cold" else s):
            raise ValueError(f"Profiler forward token 数与样本不符：{path.name}")
        forward_ms = sum(b - a for a, b in forward) / 1000
        load_ms = sum(b - a for a, b in h2d) / 1000
        overlap_ms = overlap_duration(forward, h2d) / 1000
        results.append(dict(
            trace=path.name, prefix_length=p, suffix_length=s, group=group,
            forward_kernel_busy_ms=forward_ms, h2d_kernel_busy_ms=load_ms,
            h2d_kernel_span_ms=(h2d[-1][1] - h2d[0][0]) / 1000 if h2d else 0.,
            overlap_ms=overlap_ms, h2d_without_forward_kernel_ms=load_ms - overlap_ms,
            prefill_forward_gpu_ms=sample.get(FORWARD_FIELD), kv_load_ms=sample["kv_load_ms"],
            **details,
        ))
        timelines.append((group, forward, h2d))
    if len(results) != len(samples):
        raise ValueError("trace 数量与诊断样本数量不符")
    write_json(directory / "profile_summary.json", results)
    with (directory / "profile_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    if not no_plot:
        os.environ.setdefault("MPLCONFIGDIR", str(directory.resolve() / ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(timelines), 1, figsize=(11, 3 * len(timelines)), squeeze=False)
        for ax, (group, forward, h2d) in zip(axes[:, 0], timelines):
            origin = min(a for a, _ in forward + h2d)
            for y, intervals, label, color in (
                (1, forward, "Forward kernels", "tab:blue"), (0, h2d, "KV H2D kernels", "tab:orange"),
            ):
                ax.broken_barh([((a - origin) / 1000, (b - a) / 1000) for a, b in intervals],
                               (y - .3, .6), facecolors=color, label=label)
            ax.set(title=group, yticks=[0, 1], yticklabels=["KV H2D", "Forward"], xlabel="GPU timeline (ms)")
            ax.grid(axis="x", alpha=.3)
        note = config["arguments"].get("note", "")
        title = "Profiler diagnostic: actual kernel intervals; each panel has its own time scale"
        fig.suptitle(title + ("\n" + note if note else ""))
        fig.tight_layout()
        fig.savefig(directory / "profile_timeline.png", dpi=180)
        plt.close(fig)
    print(f"Trace analysis: {directory / 'profile_summary.csv'}")


def attach_server_artifacts(directory, launch_dir, config):
    """关联客户端能读取的启动记录；远程用户可先复制记录，再显式传入目录。

    不读取服务端模型路径，不尝试 SSH/SCP，不要求两个进程共享文件系统。
    记录仅作为用户提供的来源证据，服务的实际配置仍通过 HTTP 获取。
    """
    if launch_dir is None:
        config["server_artifacts"] = None
        return
    launch_dir = Path(launch_dir).resolve()
    manifest = launch_dir / "launch.json"
    if not manifest.is_file():
        raise ValueError(f"启动记录目录中缺少 launch.json：{launch_dir}")
    config["server_artifacts"] = str(launch_dir)
    config["launch"] = json.loads(manifest.read_text())
    config["server_log_path"] = str(launch_dir / "server.log")
    if launch_dir != directory.resolve():
        for name in ("launch.json", "sglang.patch", "gpu.txt", "profile.yaml", "server-config.yaml", "profile-resolution.json"):
            if (launch_dir / name).is_file():
                shutil.copyfile(launch_dir / name, directory / name)
        if (launch_dir / "launch-source").is_dir():
            shutil.copytree(launch_dir / "launch-source", directory / "launch-source", dirs_exist_ok=True)


def run(args):
    """实验总入口：组织输入、服务检查、采样和结果保存，不处理模型计算。"""
    # 延迟导入：只有真实实验需要 tokenizer，analyze 和 self-test 不依赖它。
    from transformers import AutoTokenizer

    # 1. 创建运行目录和配置。失败或中断也要留下可诊断的记录。
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # 启动脚本可能已创建目录及日志，但不能覆盖已有的 Driver 测量结果。
    config_path = directory / "config.json"
    if config_path.exists() or (directory / "samples.csv").exists():
        raise ValueError("Run already exists; choose a new output directory or use analyze")
    config = dict(
        status="initializing",
        started_utc=datetime.now(timezone.utc).isoformat(),
        arguments=vars(args).copy(),
        phase="cold_l1_no_hicache",
    )
    # --server-artifacts 是客户端本地路径；保留旧 Docker 环境变量作为兼容默认值。
    launch_dir = args.server_artifacts
    if launch_dir is None:
        candidate = Path(os.environ.get("EXPERIMENT_LAUNCH_DIR", str(directory)))
        if (candidate / "launch.json").is_file():
            launch_dir = candidate
    config["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["packages"] = {}
    config["packages_scope"] = "client"
    for package in ("transformers", "torch", "sglang", "matplotlib"):
        try:
            config["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(config_path, config)
    # 客户端可以只有这一个文件；启动脚本的快照由服务端自行保存。
    source = directory / "source"
    source.mkdir()
    shutil.copyfile(Path(__file__), source / "run_experiment.py")
    try:
        attach_server_artifacts(directory, launch_dir, config)
        if args.telemetry_dir:
            # monotonic 时间戳不能跨主机比较，必须明确声明采样与客户端共用时钟。
            if not args.telemetry_same_host:
                raise ValueError("--telemetry-dir 需要 --telemetry-same-host；跨主机遥测不能用单调时钟直接关联")
        # 2. 使用本地 tokenizer 构造并保存精确输入；不下载模型或分词文件。
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        config["tokenizer_sha256"] = {
            name: hashlib.sha256((Path(args.tokenizer) / name).read_bytes()).hexdigest()
            for name in ("tokenizer.json", "tokenizer_config.json", "config.json")
            if (Path(args.tokenizer) / name).is_file()
        }
        text = Path(args.text_file).read_text() if args.text_file else TEXT
        cases = make_cases(tokenizer, args.prefix_lengths, args.suffix_lengths, text)
        write_json(directory / "inputs.json", dict(text=text, cases=cases))
        with (directory / "responses.jsonl").open("w") as events:
            client = Client(args.base_url, args.timeout, events)
            if args.telemetry_dir:
                client.environment = EnvironmentMonitor(Path(args.telemetry_dir))
            if args.profile:
                # 这是服务端路径，客户端不创建、不 chmod，也不假设能直接读取它。
                client.profile_dir = args.server_profile_dir
                config["server_profile_dir"] = args.server_profile_dir
            # 3. 等待模型加载及服务预热结束；启动期间的 503 不算实验失败。
            deadline = time.monotonic() + args.ready_timeout
            while True:
                try:
                    with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=3):
                        break
                except urllib.error.HTTPError as exc:
                    if exc.code != 503:
                        raise
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server did not become healthy before ready-timeout")
                time.sleep(1)
            info, _ = client.request("/server_info")
            model_info, _ = client.request("/model_info")
            config.update(
                server_info=info,
                model_info=model_info,
                model=model_info.get("model_path", info.get("model_path")),
            )
            # 4. 按实际服务配置检查输入是否合法，再进入测量。
            check_server(info, cases, args.output_length, args.groups)
            client.hicache = bool(info.get("enable_hierarchical_cache"))
            if args.gpu_timing:
                if not client.hicache:
                    raise ValueError("GPU 时间归属依赖 HiCache 空闲检查，请启用 HICACHE=1")
                # 短请求建立原生 extend 指标，并等完成后读取；不进入采样 CSV。
                client.generate(cases[0]["prefix"][:32], 1, "timing_init")
                client.wait_cache(cases[0]["prefix"][:32])
                client.gpu_timing = True
                client.load_metrics("timing_init_metrics")
            config["phase"] = "cold_l1_l2_hicache" if client.hicache else "cold_l1_no_hicache"
            common = dict(
                model=config["model"],
                dtype=info.get("dtype"),
                kv_dtype=info.get("kv_cache_dtype"),
            )
            config["status"] = "running"
            write_json(config_path, config)
            # 5. 逐个写入测量样本；预热和准备请求只出现在原始响应日志中。
            with (directory / "samples.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                run_samples(client, cases, args, writer, file, common)
            config["status"] = "complete"
    except BaseException as exc:
        config.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc),
        )
        raise
    finally:
        # 6. 无论完成、失败还是 Ctrl-C，都保存最终状态并汇总已获得的样本。
        config["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(config_path, config)
        if (directory / "samples.csv").exists():
            analyze(directory, args.no_plot)
        if launch_dir is not None:
            launch_dir = Path(launch_dir)
            if launch_dir.resolve() != directory.resolve() and (launch_dir / "server.log").is_file():
                shutil.copyfile(launch_dir / "server.log", directory / "server.log")
        print(f"Results: {directory} ({config['status']})", flush=True)


def self_test():
    """用模拟数据检查流程，不加载模型、不访问 HTTP、不产生真实性能数据。

    假 tokenizer 直接把文本编码成字节；假服务按两个输入的相同前缀计算命中。
    固定的假延迟只用于检查 CSV 和统计公式，不能用于评价推理性能。
    """
    import io
    import tempfile
    from types import SimpleNamespace

    class Tokenizer:
        """仅用于自测输入长度与分叉位置，不能代替 Qwen 的真实分词器。"""

        all_special_ids = []

        def encode(self, text, **kwargs):
            return list(text.encode())

    # 检查不同 P/S 下的首个分叉位置；这里用 8、16 等小长度加快自测。
    cases = make_cases(Tokenizer(), [8, 16], [2, 4], "fixed text")
    for c in cases:
        a, b = c["prefix"] + c["suffix"], c["prefix"] + c["warm_suffix"]
        assert next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) == len(c["prefix"])
        assert len(a) == len(b)

    class FakeClient:
        """模拟缓存状态和响应字段；不执行任何 GPU/CPU 模型推理。"""

        def __init__(self):
            self.cached = []
            self.measured = []
            self.flushes = 0

        def flush(self):
            self.cached = []
            self.flushes += 1

        def generate(self, ids, output_length, stage):
            hit = 0
            for a, b in zip(ids, self.cached):
                if a != b:
                    break
                hit += 1
            self.cached = ids.copy()
            if stage == "measure":
                self.measured.append(ids)
            return {
                "meta_info": dict(
                    prompt_tokens=len(ids),
                    completion_tokens=output_length,
                    cached_tokens=hit,
                    cached_tokens_details={"device": hit, "host": 0} if hit else None,
                    num_retractions=0,
                    finish_reason={"type": "length"},
                    e2e_latency=.002,
                    first_token_latency=.001,
                    queue_time=0,
                )
            }, 3., "fake"

    # 跑完整采样循环，检查预热不混入 CSV，且重复测量不会额外命中 suffix。
    args = SimpleNamespace(
        seed=4,
        warmup_rounds=1,
        groups=["cold", "l1"],
        output_length=1,
        repeats=3,
        max_attempts=6,
        prefix_lengths=[8],
        suffix_lengths=[2],
    )
    c = cases[0]
    client = FakeClient()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    run_samples(
        client, [c], args, writer, output,
        dict(model="test", dtype="bfloat16", kv_dtype="bfloat16"),
    )
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert len(rows) == 6 and all(r["valid"] == "True" for r in rows)
    assert client.flushes == 8  # 两组各 1 次运行预热，再加 6 次正式测量。
    assert all(ids == c["prefix"] + c["suffix"] for ids in client.measured)
    client.flush()
    good, _, _ = client.generate(c["prefix"] + c["suffix"], 1, "measure")
    assert not validate(good, 8, 2, 1, "cold")[0]
    # 逐项注入错误：错误命中、回撤、长度、时间缺失、异常结束都必须被拒绝。
    for changes in (
        {"cached_tokens": 1},
        {"cached_tokens_details": {"device": 0, "host": 1}},
        {"num_retractions": 1},
        {"prompt_tokens": 9},
        {"completion_tokens": 2},
        {"first_token_latency": None},
        {"finish_reason": {"type": "abort"}},
    ):
        bad = {"meta_info": {**good["meta_info"], **changes}}
        assert validate(bad, 8, 2, 1, "cold")[0], changes
    assert math.isclose(percentile([1, 2, 3, 4], .95), 3.85)
    # 错误服务配置应在测量前报错，而不是等长请求失败后才发现。
    server = dict(
        enable_metrics=True,
        disable_radix_cache=False,
        enable_hierarchical_cache=False,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        max_running_requests=1,
        context_length=64,
        page_size=1,
    )
    check_server(server, cases, 1)
    for changes in ({"context_length": 10}, {"page_size": 32},
                    {"enable_hierarchical_cache": True}, {"max_total_num_tokens": 8},
                    {"tp_size": 2}, {"enable_metrics": False}):
        try:
            check_server({**server, **changes}, cases, 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Bad server configuration accepted: {changes}")
    with tempfile.TemporaryDirectory() as tmp:
        # 真正写入临时 CSV 再读回汇总，检查落盘后的数值和有效性筛选。
        d = Path(tmp)
        (d / "samples.csv").write_text(output.getvalue())
        write_json(d / "config.json", dict(arguments=vars(args), status="complete"))
        analyze(d, no_plot=True)
        with (d / "summary.csv").open() as f:
            summary = list(csv.DictReader(f))
        assert len(summary) == 8 and all(r["valid"] == "3" for r in summary)
        # 极大的无效延迟要计入无效次数，但不能影响 Median 和 P95。
        bad = {**rows[0], "valid": "False", "client_latency_ms": "1000000"}
        with (d / "samples.csv").open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(bad)
        analyze(d, no_plot=True)
        with (d / "summary.csv").open() as f:
            summary = list(csv.DictReader(f))
        relevant = [r for r in summary if r["group"] == rows[0]["group"]]
        assert all(r["invalid"] == "1" and r["valid"] == "3" for r in relevant)
        assert next(r for r in relevant if r["metric"] == "client_latency_ms")["p95"] == "3.0"
    # L2 模拟只验证 Driver 的协议与指标归属，不替代真实树/allocator 测试。
    class FakeHiCacheClient(FakeClient):
        hicache = True

        def __init__(self):
            super().__init__()
            self.backed = []
            self.demoted = False
            self.counters = dict.fromkeys(LOAD_FIELDS, 0)

        def flush(self):
            super().flush()
            self.backed = []
            self.demoted = False

        def wait_cache(self, prefix_ids, *, demote=False):
            if demote:
                self.cached = []
                self.demoted = True
            def shared(ids):
                return next(
                    (i for i, (a, b) in enumerate(zip(ids, prefix_ids)) if a != b),
                    min(len(ids), len(prefix_ids)),
                )
            device, host = shared(self.cached), shared(self.backed)
            return dict(
                device_tokens=device, host_backed_tokens=host,
                host_hit_tokens=host - device, backup_complete=host == len(prefix_ids),
                device_used_tokens=len(self.cached), host_used_tokens=len(self.backed),
                kv_bytes_per_token=16,
            )

        def load_metrics(self, stage):
            return self.counters.copy()

        def generate(self, ids, output_length, stage):
            was_demoted = self.demoted
            if was_demoted:
                self.cached = self.backed.copy()
            response, elapsed, rid = super().generate(ids, output_length, stage)
            response["output_ids"] = [1] * output_length
            hit = response["meta_info"]["cached_tokens"]
            if was_demoted:
                response["meta_info"]["cached_tokens_details"] = dict(device=0, host=hit)
                self.counters["kv_load_ms"] += .25
                self.counters["kv_load_tokens"] += hit
                self.counters["kv_load_bytes"] += hit * 16
                self.counters["kv_load_ops"] += 1
            self.backed = ids.copy()
            self.demoted = False
            return response, elapsed, rid

    hi_args = SimpleNamespace(**{**vars(args), "groups": ["cold", "l1", "l2"], "repeats": 2})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    run_samples(FakeHiCacheClient(), [c], hi_args, writer, output, {})
    hi_rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert len(hi_rows) == 6 and all(r["valid"] == "True" for r in hi_rows)
    for row in hi_rows:
        expected = 8 if row["group"] == "l2" else 0
        assert int(row["kv_load_tokens"]) == expected
        assert int(row["kv_load_bytes"]) == expected * 16
        assert int(row["device_tokens_before"]) == (8 if row["group"] == "l1" else 0)

    class WrongOutputClient(FakeHiCacheClient):
        def generate(self, ids, output_length, stage):
            is_l2 = self.demoted
            response, elapsed, rid = super().generate(ids, output_length, stage)
            if is_l2 and stage == "measure":
                response["output_ids"] = [999] * output_length
            return response, elapsed, rid

    # 即使命中和传输量正确，L2 生成结果不一致也不能收为有效样本。
    wrong_args = SimpleNamespace(**{**vars(hi_args), "repeats": 1, "max_attempts": 1})
    wrong_output = io.StringIO()
    wrong_writer = csv.DictWriter(wrong_output, fieldnames=FIELDS)
    wrong_writer.writeheader()
    try:
        run_samples(WrongOutputClient(), [c], wrong_args, wrong_writer, wrong_output, {})
    except RuntimeError:
        pass
    else:
        raise AssertionError("生成结果不一致时仍完成了实验")
    wrong_rows = list(csv.DictReader(io.StringIO(wrong_output.getvalue())))
    assert any(r["valid"] == "False" and "token" in r["invalid_reason"] for r in wrong_rows)
    labels = 'cache_type="unified",tp_rank="0",pp_rank="0",dp_rank="0"'
    text = (
        f'sglang:load_back_tokens_total{{{labels},pool="kv"}} 8\n'
        f'sglang:load_back_bytes_total{{{labels}}} 128\n'
        f'sglang:load_back_duration_seconds_sum{{{labels}}} .00025\n'
        f'sglang:load_back_duration_seconds_count{{{labels}}} 1\n'
    )
    parsed = parse_load_metrics(text)
    zero = dict.fromkeys(LOAD_FIELDS, 0)
    assert parsed == dict(kv_load_ms=.25, kv_load_tokens=8, kv_load_bytes=128, kv_load_ops=1)
    assert not load_deltas(zero, parsed, 8, 16, "l2")[1]
    assert load_deltas(parsed, zero, 8, 16, "l2")[1]  # 指标重置。
    assert load_deltas(zero, parsed, 8, 16, "l1")[1]  # L1 中出现 H2D。
    assert load_deltas(zero, {**parsed, "kv_load_bytes": 64}, 8, 16, "l2")[1]
    for invalid in (text + text, "", text.replace('tp_rank="0"', 'tp_rank="1"'),
                    text.replace(' .00025', ' NaN')):
        try:
            parse_load_metrics(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("无效指标被接受")
    forward_labels = 'engine_type="unified",model_name="/model",moe_ep_rank="0",pp_rank="0",tp_rank="0"'
    forward = f'sglang:forward_execution_seconds_total{{{forward_labels},category="extend"}} .123\n'
    assert math.isclose(parse_forward_metric(forward), 123)
    for invalid in ("", forward + forward, forward.replace(".123", "NaN"),
                    forward.replace('tp_rank="0"', 'tp_rank="1"'),
                    forward.replace('category="extend"', 'category="decode"')):
        try:
            parse_forward_metric(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("无效的 GPU forward 指标被接受")
    # 区间须先取并集再算交集；并发 Kernel 不能重复累计为 GPU 忙碌时间。
    assert merge_intervals([(1, 4), (2, 3), (4, 6), (8, 9)]) == [(1, 6), (8, 9)]
    assert overlap_duration([(1, 6), (8, 9)], [(3, 8.5)]) == 3.5
    trace = [
        dict(cat="user_annotation", name="step[EXTEND bs=1 toks=128]", pid=1, tid=1, ts=100, dur=100),
        dict(cat="cuda_runtime", pid=1, tid=1, ts=110, dur=2, args=dict(correlation=7)),
        dict(cat="cuda_runtime", pid=1, tid=2, ts=110, dur=2, args=dict(correlation=9)),
        dict(cat="kernel", name="gemm", ts=150, dur=30, args=dict(device=0, correlation=7)),
        dict(cat="kernel", name="hicache_transfer_per_layer", ts=140, dur=30, args=dict(device=0, correlation=8)),
        dict(cat="kernel", name="hicache_relayout_kernel", ts=190, dur=20, args=dict(device=0, correlation=9)),
    ]
    forward, load, details = trace_intervals(trace)
    assert forward == [(150, 180)] and load == [(140, 170)]
    assert details["forward_input_tokens"] == 128 and overlap_duration(forward, load) == 20
    # 重现本轮故障：GPU0 丢失时仍保留 GPU3；实验卡自身故障必须拒绝。
    class FakeNVML:
        class NVMLError(Exception):
            pass
        NVML_TEMPERATURE_GPU, NVML_CLOCK_SM, NVML_CLOCK_MEM = 0, 0, 1
        def nvmlDeviceGetCount(self): return 4
        def nvmlDeviceGetHandleByIndex(self, index):
            if index == 0: raise self.NVMLError("GPU0 lost")
            return index
        def nvmlDeviceGetUtilizationRates(self, handle): return SimpleNamespace(gpu=0, memory=0)
        def nvmlDeviceGetMemoryInfo(self, handle): return SimpleNamespace(used=1, total=48)
        def nvmlDeviceGetComputeRunningProcesses(self, handle): return [SimpleNamespace(pid=42, usedGpuMemory=1)]
        def nvmlDeviceGetUUID(self, handle): return f"GPU-{handle}"
    sampled = sample_gpu_devices(FakeNVML(), 3)
    assert sampled["ok"] and not sampled["gpus"]["0"]["ok"]
    assert sampled["gpus"]["3"]["compute_processes"] == [dict(pid=42, used_bytes=1)]
    assert not sample_gpu_devices(FakeNVML(), 0)["ok"]
    assert not sample_gpu_devices(FakeNVML(), 9)["ok"]
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        write_json(directory / "owner.json", dict(gpu_index=3,allowed_gpu_pids=[42],scheduler_pid=42))
        base = time.monotonic_ns() - 10_000_000_000
        window = dict(measurement_start_monotonic_ns=base+1_200_000_000,
                      measurement_end_monotonic_ns=base+1_400_000_000)
        def telemetry(kind, bad_pid=False, failed=False, gap=False):
            moments = [base,base+4_000_000_000] if gap else [base+i*1_000_000_000 for i in range(4)]
            records = [dict(ok=not failed,end_monotonic_ns=t,duration_ms=2.,
                            gpus={"3":dict(compute_processes=[dict(pid=99 if bad_pid else 42)])}) for t in moments]
            (directory / f"{kind}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        telemetry("gpu");telemetry("host")
        assert EnvironmentMonitor(directory).evaluate(window)["environment_valid"]
        records = [dict(sampled, end_monotonic_ns=base+i*1_000_000_000, duration_ms=2.) for i in range(4)]
        (directory / "gpu.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        assert EnvironmentMonitor(directory).evaluate(window)["environment_valid"]
        for record in records:
            record.update(ok=False, gpus={"3":dict(ok=False, error="lost")})
        (directory / "gpu.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        assert not EnvironmentMonitor(directory).evaluate(window)["environment_valid"]
        telemetry("gpu",bad_pid=True)
        assert not EnvironmentMonitor(directory).evaluate(window)["environment_valid"]
        telemetry("gpu");telemetry("host",failed=True)
        assert not EnvironmentMonitor(directory).evaluate(window)["environment_valid"]
        telemetry("host");telemetry("gpu",gap=True)
        assert not EnvironmentMonitor(directory).evaluate(window)["environment_valid"]
        telemetry("gpu")
        reader = EnvironmentMonitor(directory)
        reader.refresh()
        record = dict(ok=True,end_monotonic_ns=base+5_000_000_000,duration_ms=1.,
                      gpus={"3":dict(compute_processes=[dict(pid=42)])})
        encoded = (json.dumps(record)+"\n").encode()
        with (directory/'gpu.jsonl').open('ab') as file:
            file.write(encoded[:30])
        reader.refresh()
        assert len(reader.records['gpu'])==4
        with (directory/'gpu.jsonl').open('ab') as file:
            file.write(encoded[30:])
        reader.refresh()
        assert len(reader.records['gpu'])==5
    class FakeEnvironment:
        def __init__(self):self.calls=0
        def wait_ready(self):pass
        def evaluate(self, window):
            self.calls += 1
            return dict(environment_valid=self.calls>1,environment_reason="injected gap" if self.calls==1 else "")
    env_client = FakeHiCacheClient()
    env_client.environment = FakeEnvironment()
    env_args = SimpleNamespace(**{**vars(hi_args),"groups":["l1"],"repeats":1,"max_attempts":3})
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=FIELDS);writer.writeheader()
    run_samples(env_client,[c],env_args,writer,output,{})
    attempts=list(csv.DictReader(io.StringIO(output.getvalue())))
    assert len(attempts)==2 and attempts[0]['cache_valid']=='True' and attempts[0]['valid']=='False'
    assert attempts[1]['valid']=='True'
    formal=summarize_rows(attempts,dict(arguments=vars(env_args)),['client_latency_ms'])
    unfiltered=summarize_rows(attempts,dict(arguments=vars(env_args)),['client_latency_ms'],'cache_valid')
    assert formal[0]['valid']==1 and formal[0]['invalid']==1 and unfiltered[0]['valid']==2
    print("PASS: cache preparation/hits, counters, trace attribution, telemetry gaps/processes, retained attempts, CSV and quantiles (mock clients only)")


def build_parser():
    """构建纯 argparse 接口；总控可复用它校验参数，不导入推理依赖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # run：只连接指定服务，部署方式与当前客户端无关。
    p = sub.add_parser("run", help="在独占 TP=1 服务上测量 Cold/L1，或开启 HiCache 后测三组")
    p.add_argument(
        "--base-url", default="http://127.0.0.1:39081", help="SGLang HTTP 服务地址"
    )
    p.add_argument("--tokenizer", required=True, help="与服务端模型匹配的本地 tokenizer 目录")
    p.add_argument("--output-dir", required=True, help="本次结果目录，不可覆盖已有测量")
    p.add_argument("--server-artifacts", help="可选：客户端可读的服务启动记录目录或其副本（含 launch.json）")
    p.add_argument(
        "--prefix-lengths", nargs="+", type=positive_length,
        default=[8192, 16384, 32768], help="历史 P 的 token 数，可写 8k 16k 32k",
    )
    p.add_argument(
        "--suffix-lengths", nargs="+", type=positive_length, default=[128],
        help="新增 S 的 token 数，可一次指定多个值",
    )
    p.add_argument(
        "--output-length", type=positive_length, default=1, help="每个测量请求生成的 token 数"
    )
    p.add_argument(
        "--groups", nargs="+", choices=["cold", "l1", "l2"], default=["cold", "l1"],
        help="测量哪些缓存组；L2 需要开启 HiCache 和实验接口",
    )
    p.add_argument(
        "--repeats", type=positive_length, default=20, help="每个 P/S/缓存组所需的有效样本数"
    )
    p.add_argument(
        "--max-attempts", type=positive_length,
        help="每组最大尝试次数，默认 repeats 的 3 倍；无效样本也消耗次数",
    )
    p.add_argument(
        "--warmup-rounds", type=positive_length, default=1,
        help="每个 P/S 组合的运行预热轮数，不计入有效样本",
    )
    p.add_argument("--seed", type=int, default=0, help="控制各组执行顺序的随机种子")
    p.add_argument("--text-file", help="固定 UTF-8 前缀文本；省略时使用内置英文笔记")
    p.add_argument("--timeout", type=positive_length, default=600, help="单次 HTTP 请求超时，单位秒")
    p.add_argument("--ready-timeout", type=positive_length, default=600, help="等待服务就绪的上限，单位秒")
    p.add_argument("--no-plot", action="store_true", help="保存 CSV 汇总但不绘图")
    p.add_argument("--profile", action="store_true", help="只在测量请求期间采集 CPU/GPU trace；结果含 Profiler 开销，仅用于诊断")
    p.add_argument("--server-profile-dir", help="Profiler 写入的服务端绝对路径；启用 --profile 时必须指定")
    p.add_argument("--gpu-timing", action="store_true", help="记录原生 extend CUDA Event 区间；服务需 FORWARD_TIMING=1，区间可能包含等待")
    p.add_argument("--note", default="", help="保存在配置和图中的实验说明，例如未缩放 RoPE 外推；不影响请求内容")
    p.add_argument("--telemetry-dir", help="宿主机采样目录；启用后关联请求窗口，监控缺口或额外 GPU 进程不计入正式有效数")
    p.add_argument("--telemetry-same-host", action="store_true", help="确认客户端与遥测进程在同一主机且共用单调时钟；远程 SSH 隧道不满足此条件")

    # analyze：只读取现有结果，适合调整图表或重新检查统计。
    p = sub.add_parser("analyze", help="从已保存样本重新汇总和绘图，无需服务或 GPU")
    p.add_argument("directory", type=Path, help="包含 config.json 和 samples.csv 的结果目录")
    p.add_argument("--no-plot", action="store_true", help="只重新生成汇总 CSV")
    sub.add_parser("self-test", help="用模拟数据自测逻辑，无需第三方依赖或 GPU")
    p = sub.add_parser("monitor", help="宿主机独立采样进程；GPU 与主机各启动一个")
    p.add_argument("--kind", choices=["gpu", "host"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--interval", type=float, default=1.)
    p = sub.add_parser("analyze-traces", help="分析三组各一次的诊断 trace，统计计算/H2D 忙碌区间与重叠")
    p.add_argument("directory", type=Path, help="完整的 --profile 运行目录")
    p.add_argument("--no-plot", action="store_true", help="仅保存 trace 汇总，不绘图")
    return parser


def normalize_run_args(parser, args):
    """CLI 和总控共用的参数校验与去重。"""
    if args.profile and (not args.server_profile_dir or not args.server_profile_dir.startswith("/")):
        parser.error("--profile 需要 --server-profile-dir 指定服务端绝对路径")
    if args.server_profile_dir and not args.profile:
        parser.error("--server-profile-dir 只能与 --profile 一起使用")
    if args.telemetry_same_host and not args.telemetry_dir:
        parser.error("--telemetry-same-host 需要 --telemetry-dir")
    for key in ("prefix_lengths", "suffix_lengths", "groups"):
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    args.max_attempts = args.max_attempts or args.repeats * 3
    if args.max_attempts < args.repeats:
        parser.error("--max-attempts must be at least --repeats")
    return args


def main():
    """解析命令行并分发子命令；CSV 字段名和现有参数名保持稳定。"""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "monitor":
        monitor(args)
    elif args.command == "analyze":
        analyze(args.directory, args.no_plot)
    elif args.command == "analyze-traces":
        analyze_traces(args.directory, args.no_plot)
    else:
        run(normalize_run_args(parser, args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
