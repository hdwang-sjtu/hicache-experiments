#!/usr/bin/env python3
"""Cold/L1 缓存实验：构造精确输入、发送请求、验证命中并分析结果。

阅读顺序：main → run → run_samples → execute_sample → validate。
  run        运行实验：准备输入 → 等待服务 → 采样 → 保存 CSV 和汇总。
  analyze    读取已有结果重新汇总，不访问模型或服务器。
  self-test  用假 tokenizer 和假服务检查脚本逻辑，不做真实模型推理。

约定：P 是可复用的历史前缀长度，S 是本次新增输入长度，S₀ 是预热用后缀。
Cold 不预热；L1 先发送 P+S₀，再发送 P+S，且 S₀ 与 S 从首个 token 分叉。
此脚本只实现阶段 1；GPU Prefill 计时和 CPU L2 恢复不在本阶段内。
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
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
# CSV 字段保持英文，方便脚本处理；含义见 README 中的“指标含义”。
FIELDS = [
    "case", "group", "attempt", "request_id", "model", "dtype", "kv_dtype",
    "prefix_length", "suffix_length", "output_length", "input_sha256",
    "prompt_tokens", "completion_tokens", "cached_tokens", "device_tokens",
    "host_tokens", "storage_tokens", "num_retractions", *TIMES,
    "forward_entry_time", "prefill_finished_time", "valid", "invalid_reason",
]


def write_json(path, value):
    """保存缩进后的 JSON，便于直接打开检查参数和原始输入。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


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
            raise RuntimeError(f"{stage} {path}: {exc} {body}") from exc
        elapsed = (time.perf_counter_ns() - started) / 1e6
        result = json.loads(raw) if parse_json else raw.decode()
        self.events.write(
            json.dumps(
                dict(
                    stage=stage,
                    path=path,
                    request_id=(payload or {}).get("rid"),
                    client_latency_ms=elapsed,
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
    expected = p if group == "l1" else 0
    checks = {
        "prompt_tokens": (meta.get("prompt_tokens"), p + s),
        "completion_tokens": (meta.get("completion_tokens"), output_length),
        "cached_tokens": (cached, expected),
        "device": (details.get("device"), expected),
        "host": (details.get("host"), 0),
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
    client.flush()
    if group == "l1":
        # 预热请求本身应是 Cold；准备失败意味着测量起点不可靠，直接终止。
        prepared, _, _ = client.generate(case["prefix"] + case["warm_suffix"], 1, "prepare")
        errors, _ = validate(prepared, p, s, 1, "cold")
        if errors:
            raise RuntimeError("Preparation failed: " + "; ".join(errors))
    # 无论哪一组，发送的测量输入都完全相同。
    result, elapsed, rid = client.generate(
        case["prefix"] + case["suffix"], output_length, stage
    )
    errors, hits = validate(result, p, s, output_length, group)
    meta = result.get("meta_info", {})
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
    )
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


def check_server(info, cases, output_length):
    """开始采样前检查服务配置、上下文长度、页对齐和可见的 KV 容量。"""
    # 当前阶段只接受关闭 HiCache 的单卡、单请求服务，确保命中归属清晰。
    required = dict(
        enable_metrics=True,
        disable_radix_cache=False,
        enable_hierarchical_cache=False,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        max_running_requests=1,
    )
    for key, expected in required.items():
        if info.get(key) != expected:
            raise ValueError(f"Phase 1 requires server {key}={expected!r}; got {info.get(key)!r}")
    if info.get("speculative_algorithm"):
        raise ValueError("Disable speculative decoding for this experiment")
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
        # 这里的 warmup 用于预热运行路径，不写入 samples.csv。
        # 它与 execute_sample 内每次 L1 都要执行的 prepare 请求不同。
        for _ in range(args.warmup_rounds):
            for group in args.groups:
                row = execute_sample(client, case, group, args.output_length, "warmup")
                if not row["valid"]:
                    raise RuntimeError("Warmup failed: " + row["invalid_reason"])
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


def analyze(directory, no_plot=False):
    """从已保存的 CSV 重新计算汇总和图表，不重新运行任何模型请求。"""
    config = json.loads((directory / "config.json").read_text())
    with (directory / "samples.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
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
                valid = [r for r in selected if r["valid"] == "True"]
                for metric in TIMES:
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
    with (directory / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
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
        # 每个 S 一张子图，横轴为 P；只画收满有效样本的组。
        # 图中文字保留英文，避免依赖容器中的中文字库。
        fig, axes = plt.subplots(
            1, len(suffixes), figsize=(6 * len(suffixes), 4), squeeze=False
        )
        for ax, s in zip(axes[0], suffixes):
            for group in config["arguments"]["groups"]:
                data = sorted(
                    [
                        r for r in summaries
                        if r["suffix_length"] == s
                        and r["group"] == group
                        and r["metric"] == "server_first_token_ms"
                        and r["complete"]
                    ],
                    key=lambda r: r["prefix_length"],
                )
                if data:
                    ax.plot(
                        [r["prefix_length"] for r in data],
                        [r["median"] for r in data],
                        "o-",
                        label=f"{group} median",
                    )
                    ax.plot(
                        [r["prefix_length"] for r in data],
                        [r["p95"] for r in data],
                        "x--",
                        label=f"{group} p95",
                    )
            ax.set(
                title=f"Suffix = {s} tokens",
                xlabel="Prefix tokens",
                ylabel="Server first-token latency (ms)",
            )
            ax.grid(alpha=.3)
            if ax.lines:
                ax.legend()
        status = config.get("status", "unknown")
        fig.suptitle(f"{config.get('model', 'model')} — run: {status}; complete groups only")
        fig.tight_layout()
        fig.savefig(directory / "latency.png", dpi=160)
        plt.close(fig)


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
    # 同一服务可以做多次实验；后续结果目录仍需关联最初的镜像、源码和日志。
    launch_dir = Path(os.environ.get("EXPERIMENT_LAUNCH_DIR", str(directory)))
    if (launch_dir / "launch.json").exists():
        config["launch"] = json.loads((launch_dir / "launch.json").read_text())
        config["server_log_path"] = str(launch_dir / "server.log")
        if launch_dir.resolve() != directory.resolve():
            shutil.copyfile(launch_dir / "launch.json", directory / "launch.json")
            if (launch_dir / "sglang.patch").exists():
                shutil.copyfile(launch_dir / "sglang.patch", directory / "sglang.patch")
    else:
        print("No launch.json found: server image/source provenance is unavailable.", file=sys.stderr)
    config["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["packages"] = {}
    for package in ("transformers", "torch", "sglang", "matplotlib"):
        try:
            config["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(config_path, config)
    # 保存实际执行的脚本，避免后来改代码后无法复现这次实验。
    source = directory / "source"
    source.mkdir()
    for name in ("run_experiment.py", "launch_server.sh"):
        shutil.copyfile(Path(__file__).with_name(name), source / name)
    try:
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
            check_server(info, cases, args.output_length)
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
    print("PASS: exact prefixes, reset/preparation, repeat isolation, invalid samples, CSV and quantiles")


def main():
    """解析命令行并分发子命令；CSV 字段名和现有参数名保持稳定。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # run：服务由 launch_server.sh 单独启动，这里只连接服务并执行实验。
    p = sub.add_parser("run", help="在独占的 TP=1 服务上测量 Cold/L1，需关闭 HiCache")
    p.add_argument(
        "--base-url", default="http://127.0.0.1:39081", help="SGLang HTTP 服务地址"
    )
    p.add_argument("--tokenizer", required=True, help="与服务端模型匹配的本地 tokenizer 目录")
    p.add_argument("--output-dir", required=True, help="本次结果目录，不可覆盖已有测量")
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
        "--groups", nargs="+", choices=["cold", "l1"], default=["cold", "l1"],
        help="测量哪些缓存组，默认同时测 Cold 和 L1",
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
    p.add_argument("--seed", type=int, default=0, help="控制两组执行顺序的随机种子")
    p.add_argument("--text-file", help="固定 UTF-8 前缀文本；省略时使用内置英文笔记")
    p.add_argument("--timeout", type=positive_length, default=600, help="单次 HTTP 请求超时，单位秒")
    p.add_argument("--ready-timeout", type=positive_length, default=600, help="等待服务就绪的上限，单位秒")
    p.add_argument("--no-plot", action="store_true", help="保存 CSV 汇总但不绘图")

    # analyze：只读取现有结果，适合调整图表或重新检查统计。
    p = sub.add_parser("analyze", help="从已保存样本重新汇总和绘图，无需服务或 GPU")
    p.add_argument("directory", type=Path, help="包含 config.json 和 samples.csv 的结果目录")
    p.add_argument("--no-plot", action="store_true", help="只重新生成汇总 CSV")
    sub.add_parser("self-test", help="用模拟数据自测逻辑，无需第三方依赖或 GPU")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "analyze":
        analyze(args.directory, args.no_plot)
    else:
        # 保留用户给出的顺序并去重，避免同一组合被无意重复测量。
        for key in ("prefix_lengths", "suffix_lengths", "groups"):
            setattr(args, key, list(dict.fromkeys(getattr(args, key))))
        args.max_attempts = args.max_attempts or args.repeats * 3
        if args.max_attempts < args.repeats:
            parser.error("--max-attempts must be at least --repeats")
        run(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
