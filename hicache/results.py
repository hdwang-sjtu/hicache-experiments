"""结果格式、来源记录与离线汇总。

这里只有 CSV/JSON 和绘图逻辑，不请求服务、不导入推理库。
字段定义由 Driver 和分析共用；旧实验 CSV 也可重新分析。
"""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys


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
    "tp", "ep", "prefill_forward_passes", "mamba_load_states",
    "load_format", "routing", "prefill_graph", "prefix_reuse", "prefill_graph_passes", "prefill_eager_passes",
    "device_tokens_before", "host_backed_tokens_before",
    "device_used_tokens_before", "host_used_tokens_before",
    "device_used_tokens_after", "host_used_tokens_after", "output_ids", "profiled", *ENV_FIELDS, "valid", "invalid_reason",
]


def mode_label(config):
    modes = config.get("benchmark_modes", {})
    label = " / ".join(str(modes.get(k, "unknown")) for k in ("load_format", "routing", "prefill_graph"))
    return label + ' / prefix_reuse=' + str(modes.get('prefix_reuse', 'unknown'))


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
    if any(r.get("environment_valid") for r in rows):
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
        fig.suptitle(title + "\n" + mode_label(config) + ("\n" + note if note else ""))
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
    fig.suptitle(title + "\n" + mode_label(config) + ("\n" + note if note else ""))
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
            fig.suptitle(f"L1 {metric} — {statistic}\n" + mode_label(config) + "\n" + config["arguments"].get("note", ""))
            fig.tight_layout()
            fig.savefig(directory / f"{metric}_{statistic}_heatmap.png", dpi=160)
            plt.close(fig)


def attach_server_artifacts(directory, launch_dir, config, *, copy_artifacts=True):
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
    # 总控结果的 server/ 已经是本轮归档的一部分，使用相对引用即可。
    # 独立客户端可能与服务不在同一主机，仍复制用户提供的启动记录。
    if not copy_artifacts:
        config["server_artifacts"] = os.path.relpath(launch_dir, directory)
        config["server_log_path"] = os.path.relpath(launch_dir / "server.log", directory)
    if copy_artifacts and launch_dir != directory.resolve():
        for name in ("launch.json", "sglang.patch", "gpu.txt", "profile.yaml", "server-config.yaml", "profile-resolution.json"):
            if (launch_dir / name).is_file():
                shutil.copyfile(launch_dir / name, directory / name)
        if (launch_dir / "launch-source").is_dir():
            shutil.copytree(launch_dir / "launch-source", directory / "launch-source", dirs_exist_ok=True)
