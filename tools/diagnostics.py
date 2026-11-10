"""可选的环境遥测与 Profiler 分析，与正式采样分开阅读。

环境覆盖使用客户端同机单调时钟；trace 使用 CUDA correlation 归属 kernel。
NVML、psutil 和 matplotlib 均按需导入，普通 HTTP 客户端不需要这些依赖。
"""
from bisect import bisect_left, bisect_right
import csv
import gzip
import json
import os
from pathlib import Path
import re
import time

from hicache.results import FORWARD_FIELD
from hicache.prepare import write_json

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("monitor")
    p.add_argument("--kind", choices=("gpu", "host"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--interval", type=float, default=1.)
    p = sub.add_parser("analyze-traces")
    p.add_argument("directory", type=Path)
    p.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.command == "monitor":
        monitor(args)
    else:
        analyze_traces(args.directory, args.no_plot)


if __name__ == "__main__":
    main()
