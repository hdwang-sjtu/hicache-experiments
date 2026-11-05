"""Driver 的模拟回归；与生产脚本分开，统一由 unittest 收集。"""
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace

from run_experiment import (make_cases, digest, run_samples, validate, check_server)
from experiment_results import FIELDS, write_json, analyze, percentile, summarize_rows
from experiment_diagnostics import (EnvironmentMonitor, sample_gpu_devices, merge_intervals,
                                    overlap_duration, trace_intervals)
from rank_metrics import RankMetrics, parse_metrics, metric_delta


def self_test(metrics_path):
    """用模拟数据检查流程，不加载模型、不访问 HTTP、不产生真实性能数据。

    假 tokenizer 直接把文本编码成字节；假服务按两个输入的相同前缀计算命中。
    固定的假延迟只用于检查 CSV 和统计公式，不能用于评价推理性能。
    """
    class Tokenizer:
        """仅用于自测输入长度，不能代替 Qwen 的真实分词器。"""

        all_special_ids = []

        def encode(self, text, **kwargs):
            return list(text.encode())

    # 检查不同 P/S 的精确长度；准备输入是独立 P。
    cases = make_cases(Tokenizer(), [8, 16], [2, 4], "fixed text")
    for c in cases:
        assert len(c["prefix"]) in (8, 16) and len(c["suffix"]) in (2, 4)
        assert digest(c["prefix"] + c["suffix"]) == c["input_sha256"]

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
                    {"tp_size": 3}, {"enable_metrics": False}):
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
    class FakeMetrics(RankMetrics):
        # 模拟服务没有异步发布，直接取当前计数；finish 仍走生产差值校验和归档。
        def snapshot(self, stage, before=None, **expected):
            return {0: self.client.counters.copy()}

    class FakeHiCacheClient(FakeClient):
        hicache = True

        def __init__(self):
            super().__init__()
            self.backed = []
            self.demoted = False
            self.counters = dict.fromkeys(("load_seconds", "kv_tokens", "load_bytes", "load_ops"), 0)
            self.rank_metrics = FakeMetrics(self, {"tp_size": 1}, False, metrics_path, gpu_timing=False)

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
                kv_bytes_per_token=16, device_free_tokens=1024-len(self.cached),
            )

        def generate(self, ids, output_length, stage):
            was_demoted = self.demoted
            if was_demoted:
                self.cached = self.backed.copy()
            response, elapsed, rid = super().generate(ids, output_length, stage)
            response["output_ids"] = [1] * output_length
            hit = response["meta_info"]["cached_tokens"]
            if was_demoted:
                response["meta_info"]["cached_tokens_details"] = dict(device=0, host=hit)
                self.counters["load_seconds"] += .00025
                self.counters["kv_tokens"] += hit
                self.counters["load_bytes"] += hit * 16
                self.counters["load_ops"] += 1
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
    parsed = parse_metrics(text, 1, 1, False, gpu_timing=False)
    zero = {0: dict.fromkeys(parsed[0], 0)}
    expectation = dict(p=8, s=2, chunk=None, state={"kv_bytes_per_token":16}, gpu_timing=False)
    assert parsed[0] == dict(load_seconds=.00025, kv_tokens=8, load_bytes=128, load_ops=1)
    metric_delta(zero, parsed, group="l2", **expectation)
    for before, after, group in ((parsed, zero, "l2"), (zero, parsed, "l1"),
                                  (zero, {0: {**parsed[0], "load_bytes": 64}}, "l2")):
        try:
            metric_delta(before, after, group=group, **expectation)
        except RuntimeError:
            pass
        else:
            raise AssertionError("错误传输量或计数倒退被接受")
    for invalid in (text + text, "", text.replace('tp_rank="0"', 'tp_rank="1"'),
                    text.replace(' .00025', ' NaN')):
        try:
            parse_metrics(invalid, 1, 1, False, gpu_timing=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("无效指标被接受")
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


class DriverRegression(unittest.TestCase):
    def test_sampling_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            self_test(Path(temp) / "rank-metrics.jsonl")


if __name__ == "__main__":
    unittest.main()
