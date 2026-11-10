#!/usr/bin/env python3
"""独立 HTTP 实验客户端：构造输入、控制缓存、验证命中并分析结果。

服务可由 Docker、serve.sh 或其他方式启动；本脚本不管理服务生命周期。
模型权重和 SGLang 只需装在服务端，客户端仅需匹配的 tokenizer 文件。

阅读顺序：main → run → run_samples → execute_sample → validate。
  run        运行实验：准备输入 → 等待服务 → 采样 → 保存 CSV 和汇总。
  analyze    读取已有结果重新汇总，不访问模型或服务器。

约定：P 是可复用的历史前缀长度，S 是本次新增输入长度。
Cold 不预热；L1/L2 单独发送 P 建立检查点，再发送 P+S。
L2 先确认 CPU 备份完成，再通过实验接口逐出 GPU。
--gpu-timing 读取原生 CUDA Event 区间；--profile 单独诊断 Kernel 执行与加载重叠。
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
import sys
import time
import urllib.error
import urllib.request
import uuid

from hicache.config import build_run_parser, normalize_run_args
from hicache.prepare import write_json
from hicache.metrics import RankMetrics, check_state, rank_states, graph_error
from hicache.metadata import snapshot_code
from hicache.results import (FIELDS, TIMES, analyze,
                                attach_server_artifacts)
from datetime import datetime, timezone
from pathlib import Path


# 固定性能测试语料。保留英文原文，避免修改注释时改变既有输入及其哈希。
TEXT = """The research assistant reads a project notebook and continues a task.
The notebook describes observations, candidate explanations, and measurements.
Each experiment uses the same inputs and records its configuration and results.
The assistant compares the evidence, checks intermediate steps, and writes down
what is known and what still needs verification before choosing the next action.
"""


def digest(value):
    """计算稳定的内容哈希，用来确认 Cold 和 L1 测量输入完全一致。"""
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()



def tile(tokens, length):
    """重复一段已分词的序列，再截取到精确长度；不重新拼接文本分词。"""
    if not tokens:
        raise ValueError("Tokenized text is empty")
    return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]


def make_cases(tokenizer, prefixes, suffixes, text):
    """为每个 P/S 组合保存独立准备的 prefix、测量 suffix 和输入哈希。"""
    unit = tokenizer.encode(text, add_special_tokens=False)
    suffix = tokenizer.encode(
        "The next task is to review the recorded observations. ",
        add_special_tokens=False,
    )
    special = set(tokenizer.all_special_ids)
    if not unit or not suffix or special.intersection(unit + suffix):
        raise ValueError("Input text must tokenize to nonempty, ordinary tokens")
    cases = []
    # 同一长度在所有组合中共享只读输入；HTTP 发送时才拼接 P+S。
    prefix_inputs = {p: tile(unit, p) for p in prefixes}
    suffix_inputs = {s: tile(suffix, s) for s in suffixes}
    for p in prefixes:
        for s in suffixes:
            measured = suffix_inputs[s]
            prefix = prefix_inputs[p]
            cases.append(
                dict(
                    id=f"p{p}_s{s}",
                    prefix=prefix,
                    suffix=measured,
                    input_sha256=digest(prefix + measured),
                )
            )
    return cases


def input_manifest(tokenizer, cases, text):
    """保存可精确重建的基序列，不把长 P 按 S 的数量重复写入磁盘。"""
    return dict(schema_version=2, text=text,
                prefix_unit=tokenizer.encode(text, add_special_tokens=False),
                suffix_unit=tokenizer.encode("The next task is to review the recorded observations. ",
                                             add_special_tokens=False),
                cases=[dict(id=c['id'], prefix_length=len(c['prefix']), suffix_length=len(c['suffix']),
                            input_sha256=c['input_sha256']) for c in cases])


class Client:
    """串行 HTTP 客户端；归档原始响应和错误，成功的 /metrics 全文仅在内存解析。"""

    def __init__(self, base_url, timeout, events):
        self.base_url, self.timeout, self.events = base_url.rstrip("/"), timeout, events
        self.hicache = False  # run() 根据实际服务配置设置，不凭命令行猜测。
        self.profile_dir = None
        self.environment = None
        self.last_request_window = None
        self.rank_metrics = None

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
        # 轮询全文高度重复；必要的逐 rank 起点、终点和差值由 RankMetrics 归档。
        if path != "/metrics":
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

    Cold 直接测量；L1 单独准备 P 的 KV/线性状态，再测量 P+S。
    清空与准备请求均不计入测量请求的延迟。每次重复都重新准备缓存。
    """
    p, s = len(case["prefix"]), len(case["suffix"])
    environment = getattr(client, "environment", None)
    if environment is not None:
        environment.wait_ready()
    client.flush()
    rank_reader = getattr(client, "rank_metrics", None)
    if group in ("l1", "l2"):
        # 预热请求本身应是 Cold；准备失败意味着测量起点不可靠，直接终止。
        if rank_reader:
            prepare_state = client.wait_cache(case["prefix"])
            check_state(prepare_state, "cold", p, rank_reader.tp)
            prepare_metrics = rank_reader.snapshot("prepare:metrics_before")
        prepared, _, prepare_rid = client.generate(case["prefix"], 1, "prepare")
        if rank_reader:
            client.wait_cache(case["prefix"])
            rank_reader.finish(prepare_metrics, "prepare", prepare_rid,
                               p=p, s=0, group="cold", state=prepare_state)
        errors, _ = validate(prepared, p, 0, 1, "cold")
        if errors:
            raise RuntimeError("Preparation failed: " + "; ".join(errors))
    before_state = None
    before_metrics = None
    if rank_reader:
        # HiCache 的状态和指标统一逐 rank 检查，单卡就是只有 rank 0。
        # 等备份完成后才允许 demote；降级前后分别验收 L1/L2 起点。
        before_state = client.wait_cache(case["prefix"])
        for rank, value in rank_states(before_state, rank_reader.tp).items():
            if value["device_free_tokens"] + value["device_used_tokens"] < p + s + output_length:
                raise RuntimeError(f"rank={rank} 实际 GPU KV 容量不足")
        check_state(before_state, "cold" if group == "cold" else "l1", p, rank_reader.tp)
        if group == "l2":
            before_state = client.wait_cache(case["prefix"], demote=True)
            check_state(before_state, "l2", p, rank_reader.tp)
        if environment is not None:
            environment.wait_ready()
        before_metrics = rank_reader.snapshot("metrics_before")
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
        extra = rank_reader.finish(before_metrics, stage, rid,
                                   p=p, s=s, group=group, state=before_state)
        if (stage in ('measure', 'warmup') and getattr(rank_reader, 'gpu_timing', False)
                and getattr(rank_reader, 'last_deltas', None)):
            error = graph_error(rank_reader.last_deltas, getattr(client, 'prefill_graph', None))
            if error:
                errors.append(error)
        extra.update(tp=rank_reader.tp, ep=rank_reader.ep)
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
    # 单请求使各 rank 的累计指标差值可以归属到当前测量请求。
    required = dict(
        enable_metrics=True,
        disable_radix_cache=False,
        pp_size=1,
        dp_size=1,
        max_running_requests=1,
    )
    for key, expected in required.items():
        if info.get(key) != expected:
            raise ValueError(f"Experiment requires server {key}={expected!r}; got {info.get(key)!r}")
    tp, ep = info.get("tp_size"), info.get("ep_size", 1)
    if tp not in (1, 2, 4) or ep not in (1, tp):
        raise ValueError("实验支持 TP=1/2/4，EP=1 或 EP=TP")
    if info.get("moe_dp_size", 1) != 1 or info.get("enable_dp_attention", False):
        raise ValueError("实验要求关闭 MoE DP 和 attention DP")
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
        if getattr(client, "rank_metrics", None) and "cold" not in args.groups:
            reference = execute_sample(client, case, "cold", args.output_length, "reference")
            if not reference["valid"]:
                raise RuntimeError("Cold reference failed: " + reference["invalid_reason"])
            reference_output = reference["output_ids"]
        # 这里的 warmup 用于预热运行路径，不写入 samples.csv。
        # 它与 execute_sample 内每次 L1 都要执行的 prepare 请求不同。
        for _ in range(args.warmup_rounds):
            for group in args.groups:
                row = execute_sample(client, case, group, args.output_length, "warmup")
                if not row["valid"]:
                    raise RuntimeError("Warmup failed: " + row["invalid_reason"])
                if reference_output is None:
                    reference_output = row["output_ids"]
                elif row["output_ids"] != reference_output:
                    raise RuntimeError("Warmup Cold/L1/L2 输出 token 不一致")
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
                if '[Graph]' in row['invalid_reason']:
                    raise RuntimeError(f"{case['id']}/{group}: {row['invalid_reason']}")
                valid[group] += int(row["valid"])
                print(
                    f"{case['id']} {group}: valid {valid[group]}/{args.repeats}, "
                    f"client={row['client_latency_ms']:.2f} ms {row['invalid_reason']}",
                    flush=True,
                )


def run(args):
    """实验总入口：组织输入、服务检查、采样和结果保存，不处理模型计算。"""
    # 延迟导入：只有真实实验需要 tokenizer，离线分析不依赖它。
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
    # --server-artifacts 是客户端本地路径；Docker 启动器也可通过环境提供它。
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
    # 客户端保存 Driver 及指标模块；启动脚本的快照由服务端自行保存。
    # 总控已经从冻结的 code/ 启动本文件；独立运行才需要自己的源码副本。
    # Docker 将同一 code/ 同时挂到 /experiment 和 /results/code，比较文件身份
    # 而非路径字符串，才能让两种 runtime 使用相同的归档规则。
    frozen_source = directory.parent / "code/hicache/benchmark.py"
    managed = frozen_source.is_file() and frozen_source.samefile(__file__)
    source = directory.parent / "code" if managed else directory / "source"
    if not managed:
        source.mkdir()
        snapshot_code(source)
    config["source_directory"] = os.path.relpath(source, directory)
    try:
        attach_server_artifacts(directory, launch_dir, config, copy_artifacts=not managed)
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
        write_json(directory / "inputs.json", input_manifest(tokenizer, cases, text))
        with (directory / "responses.jsonl").open("w") as events:
            client = Client(args.base_url, args.timeout, events)
            if args.telemetry_dir:
                from tools.diagnostics import EnvironmentMonitor
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
            graph_config = info.get('cuda_graph_config') or {}
            actual_graph = graph_config.get('prefill', {}).get('backend', info.get('cuda_graph_backend_prefill'))
            requested_graph = getattr(args, 'prefill_graph', None)
            if requested_graph and actual_graph != requested_graph:
                raise ValueError(f"服务 prefill Graph 配置不符：{actual_graph!r} != {requested_graph!r}")
            client.prefill_graph = requested_graph or actual_graph
            routing_env = config.get('launch', {}).get('profile', {}).get('environment', {})
            routing = ('round-robin' if routing_env.get('SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS') == '1'
                       else 'original' if routing_env.get('SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS') == '0'
                       else 'unknown')
            config['benchmark_modes'] = dict(load_format=info.get('load_format', 'unknown'),
                                             routing=routing, prefill_graph=client.prefill_graph or 'unknown',
                                             planned_graph_buckets=info.get('cuda_graph_bs_prefill'))
            model_config_path = Path(args.tokenizer) / "config.json"
            model_config = json.loads(model_config_path.read_text()) if model_config_path.is_file() else {}
            text_config = model_config.get("text_config", model_config)
            hybrid = "qwen3_5" in text_config.get("model_type", "")
            if hybrid or info["tp_size"] > 1:
                if not args.gpu_timing or args.output_length != 1:
                    raise ValueError("混合/多卡实验要求 GPU timing 和单 token 输出")
                chunk = info.get("chunked_prefill_size")
                grid = math.lcm(info["page_size"], text_config.get("mamba_chunk_size", 64))
                if (not isinstance(chunk, int) or chunk <= 0 or info.get("enable_dynamic_chunking")
                        or max(args.suffix_lengths) > chunk):
                    raise ValueError("需要固定正数 chunk，且 S 不大于 chunk")
                if hybrid and (info.get("mamba_radix_cache_strategy") != "extra_buffer"
                               or any(p % grid for p in args.prefix_lengths) or chunk % grid):
                    raise ValueError("混合缓存需要 extra_buffer 及按实际 checkpoint 网格对齐的 P/chunk")
                if info["tp_size"] > 1 and not info.get("enable_metrics_for_all_schedulers"):
                    raise ValueError("多卡实验需要所有 scheduler 的指标")
                capacities = [info.get("max_total_num_tokens")] + [v.get("memory_usage", {}).get("token_capacity") for v in info.get("internal_states", [])]
                if not capacities or any(not isinstance(v, int) or v < max(args.prefix_lengths)+max(args.suffix_lengths)+1 for v in capacities):
                    raise ValueError(f"实际 KV 容量不足或缺失：{capacities}")
            client.hicache = bool(info.get("enable_hierarchical_cache"))
            if args.gpu_timing:
                if not client.hicache:
                    raise ValueError("GPU 时间归属依赖 HiCache 空闲检查，请启用 HICACHE=1")
                # 用完整 P 建立原生 extend 指标，并等完成后读取；不进入采样 CSV。
                client.flush()
                client.generate(cases[0]["prefix"], 1, "timing_init")
                client.wait_cache(cases[0]["prefix"])
            if client.hicache:
                client.rank_metrics = RankMetrics(client, info, hybrid, directory / "rank-metrics.jsonl",
                                                 gpu_timing=args.gpu_timing)
                if args.gpu_timing:
                    client.rank_metrics.snapshot("timing_init_metrics")
            config["phase"] = "cold_l1_l2_hicache" if client.hicache else "cold_l1_no_hicache"
            common = dict(
                model=config["model"],
                dtype=info.get("dtype"),
                kv_dtype=info.get("kv_cache_dtype"),
                **{key: config['benchmark_modes'][key] for key in ('load_format', 'routing', 'prefill_graph')},
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
        if launch_dir is not None and not managed:
            launch_dir = Path(launch_dir)
            if launch_dir.resolve() != directory.resolve() and (launch_dir / "server.log").is_file():
                shutil.copyfile(launch_dir / "server.log", directory / "server.log")
        print(f"Results: {directory} ({config['status']})", flush=True)



def build_parser():
    """构建纯 argparse 接口；总控可复用它校验参数，不导入推理依赖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", parents=[build_run_parser()[0]],
                   help="在独占服务上测量 Cold/L1/L2，支持 TP=1/2/4")

    # analyze：只读取现有结果，适合调整图表或重新检查统计。
    p = sub.add_parser("analyze", help="从已保存样本重新汇总和绘图，无需服务或 GPU")
    p.add_argument("directory", type=Path, help="包含 config.json 和 samples.csv 的结果目录")
    p.add_argument("--no-plot", action="store_true", help="只重新生成汇总 CSV")
    return parser



def main():
    """解析命令行并分发子命令；CSV 字段名和现有参数名保持稳定。"""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        analyze(args.directory, args.no_plot)
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
