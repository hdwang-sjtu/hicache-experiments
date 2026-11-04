#!/usr/bin/env python3
"""实例内一键任务：准备固定源码/模型 → 开服务 → 测试 → 校验结果 → 清理服务。"""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from prepare_runtime import cache_lock, write_json
from rank_metrics import metric_delta
import run_experiment as driver
import serve_config

PROJECT = Path(__file__).resolve().parent
# 路径与服务地址由总控掌握；诊断 Profiler/遥测仍使用独立 Driver。
TEST_OPTIONS = {
    "prefix-lengths", "suffix-lengths", "groups", "output-length", "repeats", "max-attempts",
    "warmup-rounds", "seed", "timeout", "ready-timeout", "no-plot", "gpu-timing", "note", "text-file",
}
CODE_FILES = ("run_all.py", "prepare_runtime.py", "run_experiment.py", "serve_config.py",
              "launch_metadata.py", "serve.sh", "launch_server.sh", "rank_metrics.py")


def named(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("名称只能包含字母、数字、点、下划线和连字符")
    return name


def load_experiment(name, project=PROJECT):
    path = project / "configs/experiments" / (named(name) + ".yaml")
    if not path.is_file():
        raise ValueError(f"没有实验配置 {name!r}；用 --list-experiments 查看名称")
    data = serve_config.yaml_module().safe_load(path.read_text())
    if not isinstance(data, dict) or set(data) != {"serve", "model", "test"}:
        raise ValueError("实验配置必须包含且仅包含 serve、model、test")
    named(data["serve"])
    model = data["model"]
    if (not isinstance(model, dict) or set(model) != {"repo", "revision"}
            or not isinstance(model["repo"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", str(model["revision"]))):
        raise ValueError("model 需要 repo 和固定的完整 revision SHA")
    if not isinstance(data["test"], dict) or set(data["test"]) - TEST_OPTIONS:
        raise ValueError(f"test 仅支持这些 Driver 参数：{sorted(TEST_OPTIONS)}")
    if not {"prefix-lengths", "suffix-lengths", "groups", "repeats"} <= data["test"].keys():
        raise ValueError("实验需显式指定 prefix-lengths、suffix-lengths、groups、repeats")
    return path, data


def test_arguments(settings):
    """复用 Driver 的 argparse 类型；不复制它的长度解析和默认值。"""
    parser = driver.build_parser()
    run_parser = parser._subparsers._group_actions[0].choices["run"]
    argv = []
    for key, value in settings.items():
        option = "--" + key
        action = run_parser._option_string_actions[option]
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError(f"{key} 必须是 true/false")
            if value:
                argv.append(option)
        elif action.nargs == "+":
            if not isinstance(value, list) or not value:
                raise ValueError(f"{key} 必须是非空列表")
            argv.extend([option, *map(str, value)])
        else:
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                raise ValueError(f"{key} 必须是标量")
            argv.extend([option, str(value)])
    try:
        args = parser.parse_args(["run", "--tokenizer", "/placeholder", "--output-dir", "/placeholder", *argv])
        driver.normalize_run_args(parser, args)
    except SystemExit as exc:
        raise ValueError("实验配置中的 Driver 参数无效") from exc
    return argv, args


def ensure_port_free(host, port):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        # 允许上一轮已关闭连接的 TIME_WAIT；仍不能绑定正在监听的服务。
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"服务端口不可用 {host}:{port}；不复用或停止已有服务") from exc


def verify_results(directory, expected):
    import csv
    required = ["config.json", "inputs.json", "responses.jsonl", "samples.csv", "summary.csv"]
    if not expected.no_plot:
        required.append("latency.png")
        if expected.gpu_timing:
            required.append("timing.png")
    for name in required:
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
            raise RuntimeError(f"缺少结果文件：{name}")
    config = json.loads((directory / "config.json").read_text())
    if config.get("status") != "complete":
        raise RuntimeError("Driver 没有报告 complete")
    wanted = {(p, s, group): expected.repeats for p in expected.prefix_lengths
              for s in expected.suffix_lengths for group in expected.groups}
    counts = Counter()
    records = {}
    if expected.gpu_timing:
        path = directory / "rank-metrics.jsonl"
        if not path.is_file():
            raise RuntimeError("缺少逐 rank 指标归档")
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["request_id"] in records:
                raise RuntimeError("重复的逐 rank 请求记录")
            records[record["request_id"]] = record
        required.append("rank-metrics.jsonl")
    with (directory / "samples.csv").open() as file:
        for row in csv.DictReader(file):
            if row["valid"] == "True":
                if expected.gpu_timing:
                    record = records[row["request_id"]]
                    before = {int(k): v for k, v in record["before"].items()}
                    after = {int(k): v for k, v in record["after"].items()}
                    if set(before) != set(range(record["tp"])):
                        raise RuntimeError("归档缺少 TP rank")
                    delta = metric_delta(before, after, **record["expected"])
                    value = max(v["seconds"] for v in delta.values()) * 1000
                    if not math.isclose(value, float(row["prefill_forward_gpu_ms"])):
                        raise RuntimeError("CSV 与逐 rank 计时不符")
                counts[(int(row["prefix_length"]), int(row["suffix_length"]), row["group"])] += 1
    if dict(counts) != wanted:
        raise RuntimeError(f"有效样本未满足实验配置：{dict(counts)}")
    return dict(valid_samples=sum(counts.values()), combinations=len(counts), files=required)


class TaskInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"收到信号 {signal.Signals(signum).name}")


class MemoryMonitor:
    """Sample startup and request peaks; sampled NVML values are not allocation maxima."""
    def __init__(self, root, state):
        self.root, self.state = root, state
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.peaks = {}

    def run(self):
        with (self.root / "gpu-memory.jsonl").open("a", buffering=1) as log:
            while not self.stop.is_set():
                record = dict(utc=datetime.now(timezone.utc).isoformat(), phase=self.state["phase"])
                try:
                    output = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                         "--format=csv,noheader,nounits"], text=True, stderr=subprocess.PIPE, timeout=5)
                    gpus = []
                    for line in output.splitlines():
                        index, uuid_value, used, total, util = [v.strip() for v in line.split(",")]
                        value = dict(index=int(index), uuid=uuid_value, used_mib=int(used), total_mib=int(total), utilization=int(util))
                        gpus.append(value)
                        self.peaks[uuid_value] = max(self.peaks.get(uuid_value, 0), value["used_mib"])
                    record["gpus"] = gpus
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    record["error"] = str(exc)
                log.write(json.dumps(record) + "\n")
                self.stop.wait(.5)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=7)
        self.state["sampled_gpu_peak_mib"] = self.peaks


class Job:
    """只编排本次拥有的子进程/容器；不连接已有服务或调用云平台管理 API。"""

    def __init__(self, args, project=PROJECT, environ=None):
        self.project = Path(project)
        self.args = args
        self.env = dict(os.environ if environ is None else environ)
        self.experiment_path, self.experiment = load_experiment(args.experiment, self.project)
        for key in ("prefix_lengths", "suffix_lengths", "groups", "repeats", "no_plot"):
            value = getattr(args, key, None)
            if value is not None:
                self.experiment["test"][key.replace("_", "-")] = value
        self.test_argv, self.test = test_arguments(self.experiment["test"])
        self.lock = json.loads((self.project / "configs/runtime.lock.json").read_text())
        if getattr(args, "image", None):
            self.lock["image"] = args.image
        if getattr(args, "source_repository", None):
            self.lock["sglang_repository"] = args.source_repository
        patch = self.project / self.lock["patch"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != self.lock["patch_sha256"]:
            raise ValueError("版本锁中的补丁哈希不匹配")
        self.cache = Path(self.env.get("HICACHE_CACHE_DIR", str(self.project / ".cache"))).expanduser().resolve()
        model_root = Path(self.env.get("MODEL_ROOT", str(self.cache / "models"))).expanduser().resolve()
        # 版本化目录避免覆盖手工准备的同名模型或将不同 revision 混在一起。
        _, _, profile = serve_config.read_profile(self.experiment["serve"], self.project / "configs/serve")
        local = getattr(args, "model_dir", None) or self.env.get("MODEL_DIR")
        self.local_model = bool(local)
        self.model = (Path(local).expanduser().resolve() if local else
                      model_root / profile["model"] / self.experiment["model"]["revision"])
        self.env["MODEL_DIR"] = str(self.model)
        for key, env_key in dict(tp="TP", ep="EP", context_length="CONTEXT_LENGTH",
                                max_total_tokens="MAX_TOTAL_TOKENS", chunked_prefill_size="CHUNKED_PREFILL_SIZE",
                                mem_fraction="MEM_FRACTION", hicache_size="HICACHE_SIZE", gpu="GPU", port="PORT").items():
            if getattr(args, key, None) is not None:
                self.env[env_key] = str(getattr(args, key))
        self.resolution = serve_config.resolve(self.experiment["serve"], args.runtime, [], self.env,
                                               self.project / "configs/serve")
        server = self.resolution["server"]
        hybrid = "mamba-radix-cache-strategy" in server
        if hybrid:
            required = max(self.test.prefix_lengths) + max(self.test.suffix_lengths) + self.test.output_length
            defaults = {"context-length": max(16384, ((required + 2047) // 2048) * 2048)}
            defaults["max-total-tokens"] = max(32768, defaults["context-length"])
            for key, value in defaults.items():
                variable = "CONTEXT_LENGTH" if key == "context-length" else "MAX_TOTAL_TOKENS"
                if variable not in self.env:
                    server[key] = max(server.get(key) or 0, value)
            if "CHUNKED_PREFILL_SIZE" not in self.env:
                server["chunked-prefill-size"] = min(server["context-length"], 32768)
            if (self.test.output_length != 1 or not self.test.gpu_timing
                    or not server.get("enable-hierarchical-cache")):
                raise ValueError("混合缓存实验要求 HiCache、GPU timing 和单 token 输出")
            if min(server["context-length"], server["max-total-tokens"]) < required:
                raise ValueError("实际配置容量必须至少为 P+S+1")
            chunk = server["chunked-prefill-size"]
            if chunk <= 0 or chunk > server["context-length"] or chunk % 256:
                raise ValueError("chunk 必须为 256 的正整数倍且不超过 context")
            if max(self.test.suffix_lengths) > chunk:
                raise ValueError("S 超过 chunk；L1/L2 后缀要求一次 Forward")
            if any(p % 256 for p in self.test.prefix_lengths):
                raise ValueError("Qwen3.5 前缀需按 256 对齐")
            server["max-prefill-tokens"] = server["context-length"]
        tp, ep = server.get("tp-size", 1), server.get("ep-size", 1)
        if tp not in (1, 2, 4) or ep not in (1, tp):
            raise ValueError("支持 TP=1/2/4、EP=1 或 EP=TP")
        self.host, self.port = server.get("host", "127.0.0.1"), int(server.get("port", 39081))
        if self.host not in ("127.0.0.1", "localhost", "0.0.0.0", "::", "::1") or not 0 < self.port < 65536:
            raise ValueError("一键任务只启动当前主机服务，请使用 loopback 或通配监听地址及有效端口")
        client_host = "[::1]" if ":" in self.host else "127.0.0.1"
        self.url = f"http://{client_host}:{self.port}"
        # 尽早检查可判断的参数，避免下载完模型才发现 8K 输入配了 4K 上下文。
        for p in self.test.prefix_lengths:
            for s in self.test.suffix_lengths:
                if p + s + self.test.output_length > int(server["context-length"]):
                    raise ValueError("实验输入和输出超过服务配置的总上下文")
        if "l2" in self.test.groups and not server.get("enable-hierarchical-cache"):
            raise ValueError("L2 实验需要开启 HiCache")
        if self.test.gpu_timing and not serve_config.boolean(self.resolution["environment"].get("SGLANG_ENABLE_METRICS_DEVICE_TIMER", "0"), "GPU timing"):
            raise ValueError("实验要求 GPU 计时，但服务配置没有启用它")
        self.run_id = named(args.run_id or self.env.get("RUN_ID") or
                            (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.experiment))
        self.root = Path(self.env.get("RESULTS_DIR", str(self.project / "results"))).expanduser().resolve() / self.run_id
        self.code = self.root / "code"
        self.python = self.env.get("PYTHON_BIN", sys.executable)
        self.identity = "hicache-job-" + uuid.uuid4().hex
        self.env.update(RESULTS_DIR=str(self.root), RUN_ID="server", MODEL_DIR=str(self.model),
                        CONTAINER_NAME=self.identity, IMAGE=self.lock["image"], PYTHONUNBUFFERED="1",
                        GIT_TERMINAL_PROMPT="0", PYTHON_BIN=self.python)
        # 用绝对 Python 路径执行总控时，宿主机包装脚本也使用该解释器所在环境。
        self.env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + self.env.get("PATH", os.defpath)
        self.env["NO_PROXY"] = self.env.get("NO_PROXY", "") + ",127.0.0.1,localhost,::1"
        self.processes = []
        self.active_processes = []
        self.threads = []
        self.server = None
        self.state = dict(status="initializing", phase="initializing", run_id=self.run_id,
                          experiment=args.experiment, runtime=args.runtime,
                          started_utc=datetime.now(timezone.utc).isoformat(),
                          expected_image=self.lock["image"], image_verified=False,
                          service_identity=self.identity, base_url=self.url)

    def phase(self, name):
        self.state["phase"] = name
        write_json(self.root / "run.json", self.state)
        print(f"[{name}] {self.root}", flush=True)

    def snapshot(self):
        self.code.mkdir()
        self.state["code_sha256"] = {}
        for name in CODE_FILES:
            content = (self.project / name).read_bytes()
            (self.code / name).write_bytes(content)
            self.state["code_sha256"][name] = hashlib.sha256(content).hexdigest()
        profiles = self.code / "configs/serve"
        profiles.mkdir(parents=True)
        # Freeze resolved values, including CLI overrides and adaptive capacity.
        frozen = dict(self.resolution["server"])
        frozen.pop("model-path", None)
        _, _, profile = serve_config.read_profile(self.experiment["serve"], self.project / "configs/serve")
        profile.update(sglang=frozen, env=self.resolution["environment"])
        (profiles / (self.experiment["serve"] + ".yaml")).write_text(
            serve_config.yaml_module().safe_dump(profile, allow_unicode=True, sort_keys=False))
        (self.root / "serve-original.yaml").write_text(self.resolution["source_text"])
        write_json(self.root / "resolved-experiment.json", self.experiment)
        (self.root / "experiment.yaml").write_bytes(self.experiment_path.read_bytes())
        write_json(self.root / "runtime.lock.json", self.lock)
        shutil.copyfile(self.project / self.lock["patch"], self.root / "sglang.patch")
        if "text-file" in self.experiment["test"]:
            path = Path(self.experiment["test"]["text-file"]).expanduser()
            if not path.is_absolute():
                path = self.experiment_path.parent / path
            shutil.copyfile(path, self.root / "input.txt")
            position = self.test_argv.index("--text-file") + 1
            self.test_argv[position] = "/results/input.txt" if self.args.runtime == "docker" else str(self.root / "input.txt")
        self.state["status"] = "running"
        self.phase("preflight")

    def start(self, command, log_name):
        process = subprocess.Popen(command, cwd=self.code, env=self.env, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        self.processes.append(process)
        self.active_processes.append(process)
        self.state.setdefault("processes", []).append(dict(pid=process.pid, command=command, log=log_name))
        write_json(self.root / "run.json", self.state)

        def copy_output():
            with process.stdout, (self.root / log_name).open("a", buffering=1) as file:
                for line in process.stdout:
                    file.write(line)
                    print(line, end="", flush=True)
        thread = threading.Thread(target=copy_output, daemon=True)
        thread.start()
        self.threads.append(thread)
        return process

    def wait(self, process, timeout=None, watch_server=False):
        started = time.monotonic()
        while process.poll() is None:
            if watch_server and self.server.poll() is not None:
                raise RuntimeError(f"测试途中服务退出：{self.server.returncode}")
            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError(f"子进程超时（{timeout} 秒），见任务日志")
            time.sleep(.1)
        self.stop_process(process, grace=0)
        if process.returncode:
            raise RuntimeError(f"子进程退出码 {process.returncode}，见任务日志")

    def command(self, argv, log="prepare.log", timeout=1800):
        self.wait(self.start(argv, log), timeout)

    def prepare(self):
        if self.args.runtime == "docker":
            self.phase("preparing_image")
            if subprocess.run(["docker", "image", "inspect", self.lock["image"]], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=30).returncode:
                self.command(["docker", "pull", self.lock["image"]])
            self.state["actual_image_id"] = subprocess.check_output(
                ["docker", "image", "inspect", self.lock["image"], "--format", "{{.Id}}"], text=True, timeout=30).strip()
            self.state["image_verified"] = True
        else:
            # 使用实际运行的 Python 检查依赖；不假设实例一定来自期望的镜像。
            modules = ["torch", "transformers", "huggingface_hub"]
            if not self.test.no_plot:
                modules.append("matplotlib")
            probe = "import importlib.util; names=" + repr(modules) + "; missing=[x for x in names if importlib.util.find_spec(x) is None]; assert not missing, 'Missing dependencies: '+str(missing)"
            self.command([self.python, "-c", probe])
        self.phase("preparing_source")
        self.command([sys.executable, str(self.code / "prepare_runtime.py"), "source", "--cache", str(self.cache / "sglang"),
                      "--lock", str(self.root / "runtime.lock.json"), "--patch", str(self.root / "sglang.patch"),
                      "--output", str(self.root / "source.json")])
        self.state["source"] = json.loads((self.root / "source.json").read_text())
        self.env["SGLANG_DIR"] = self.state["source"]["path"]
        self.phase("preparing_model")
        model_command = [sys.executable, str(self.code / "prepare_runtime.py"),
                         "inspect-model" if self.local_model else "model", "--directory", str(self.model),
                         "--output", str(self.root / "model.json")]
        if not self.local_model:
            model_command += ["--repo", self.experiment["model"]["repo"],
                              "--revision", self.experiment["model"]["revision"]]
        self.command(model_command, timeout=7200)
        self.state["model"] = json.loads((self.root / "model.json").read_text())

    def wait_ready(self):
        deadline = time.monotonic() + self.test.ready_timeout
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            if self.server.poll() is not None:
                raise RuntimeError(f"服务启动失败，退出码 {self.server.returncode}")
            try:
                with opener.open(self.url + "/health", timeout=2) as response:
                    healthy = response.status == 200
                if healthy:
                    with opener.open(self.url + "/server_info", timeout=2) as response:
                        info = json.load(response)
                    # 使用原生 served-model-name 确认身份，不会误测端口上恰好出现的其他服务。
                    if info.get("served_model_name") != self.identity:
                        raise RuntimeError("就绪端口上的服务不属于本任务，拒绝发送测试请求")
                    write_json(self.root / "server-ready.json", info)
                    return
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"服务在 {self.test.ready_timeout} 秒内未就绪")
            time.sleep(.2)

    def driver_command(self):
        if self.args.runtime == "docker":
            cid = self.container_id()
            if not cid:
                raise RuntimeError("没有本次容器 ID")
            return ["docker", "exec", "--user", f"{os.getuid()}:{os.getgid()}", cid,
                    "python3", "/experiment/run_experiment.py", "run", "--base-url", self.url,
                    "--tokenizer", "/model", "--output-dir", "/results/samples",
                    "--server-artifacts", "/results/server", *self.test_argv]
        return [self.python, str(self.code / "run_experiment.py"), "run", "--base-url", self.url,
                "--tokenizer", str(self.model), "--output-dir", str(self.root / "samples"),
                "--server-artifacts", str(self.root / "server"), *self.test_argv]

    def container_id(self):
        path = self.root / "server/container.id"
        if not path.is_file():
            return None
        cid = path.read_text().strip()
        if not re.fullmatch(r"[a-f0-9]{12,64}", cid):
            raise RuntimeError("本次容器 ID 记录无效")
        return cid

    def cleanup(self):
        # 子进程各有独立进程组；信号不会发给用户的 shell 或其他实验。
        errors = []
        if self.args.runtime == "docker":
            try:
                cid = self.container_id()
                if cid:
                    try:
                        subprocess.run(["docker", "stop", "--time", "15", cid], capture_output=True, timeout=30)
                    finally:
                        removed = subprocess.run(["docker", "rm", "--force", cid], capture_output=True, text=True, timeout=30)
                        if removed.returncode and "No such" not in removed.stderr:
                            raise RuntimeError("无法确认本次容器已删除：" + removed.stderr.strip())
            except Exception as exc:
                errors.append(str(exc))
        # 已结束的准备进程已及时退休，不能在数小时后再向旧 PID 发信号。
        for process in list(reversed(self.active_processes)):
            try:
                self.stop_process(process)
            except Exception as exc:
                errors.append(str(exc))
        for thread in self.threads:
            thread.join(timeout=5)
        # Driver 可能在服务停止前复制过日志，最后再保存完整版本。
        if (self.root / "samples").is_dir() and (self.root / "server/server.log").is_file():
            shutil.copyfile(self.root / "server/server.log", self.root / "samples/server.log")
        if errors:
            raise RuntimeError("; ".join(errors))

    def stop_process(self, process, grace=15):
        if process not in self.active_processes:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        self.active_processes.remove(process)

    def execute(self):
        # 创建目录失败不会触碰旧任务，后续异常才进入本次任务的清理路径。
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(exist_ok=False)
        locks = ExitStack()
        previous_handlers = {}
        monitor = None
        exit_code = 1
        def interrupted(signum, _frame):
            raise TaskInterrupted(signum)
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, interrupted)
            self.snapshot()
            locks.enter_context(cache_lock(self.cache / "locks" / f"port-{self.port}.lock"))
            ensure_port_free(self.host, self.port)
            self.prepare()
            ensure_port_free(self.host, self.port)
            self.phase("starting_server")
            monitor = MemoryMonitor(self.root, self.state)
            monitor.thread.start()
            launcher = "serve.sh" if self.args.runtime == "native" else "launch_server.sh"
            self.server = self.start(["bash", str(self.code / launcher), self.experiment["serve"],
                                      "--served-model-name", self.identity], "server-launch.log")
            self.wait_ready()
            self.phase("testing")
            self.wait(self.start(self.driver_command(), "driver.log"), watch_server=True)
            self.phase("validating")
            self.state["validation"] = verify_results(self.root / "samples", self.test)
            exit_code = 0
        except TaskInterrupted as exc:
            self.state.update(status="interrupted", error=str(exc), failed_phase=self.state["phase"])
            exit_code = 128 + exc.signum
        except Exception as exc:
            self.state.update(status="failed", error=str(exc), failed_phase=self.state["phase"])
            print(f"任务失败：{exc}", file=sys.stderr, flush=True)
        finally:
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
            try:
                self.phase("cleanup")
                self.cleanup()
            except Exception as exc:
                self.state["cleanup_error"] = str(exc)
                if exit_code == 0:
                    self.state.update(status="failed", failed_phase="cleanup")
                    exit_code = 1
            if monitor is not None:
                monitor.close()
            locks.close()
            if exit_code == 0:
                self.state["status"] = "complete"
            self.state.update(phase="finished", exit_code=exit_code, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(self.root / "run.json", self.state)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            print(f"任务 {self.state['status']}，结果：{self.root}", flush=True)
        return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", help="configs/experiments 下的配置名")
    parser.add_argument("--runtime", choices=["native", "docker"], default="native")
    parser.add_argument("--run-id", help="本轮独立目录名，不能覆盖已有任务")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置和显示任务，不联网、不启动、不写目录")
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--model-dir", help="只读使用现有本地模型，不下载或重写权重")
    parser.add_argument("--image", help="Docker 镜像覆盖；记录实际 image ID")
    parser.add_argument("--source-repository", help="固定基线的 Git 来源，可使用本地仓库离线准备")
    parser.add_argument("--gpu", help="GPU 编号列表，例如 0,1,2,3")
    parser.add_argument("--port", type=int)
    parser.add_argument("--tp", type=int, choices=[1, 2, 4])
    parser.add_argument("--ep", type=int, choices=[1, 2, 4])
    parser.add_argument("--prefix-lengths", nargs="+", type=driver.positive_length)
    parser.add_argument("--suffix-lengths", nargs="+", type=driver.positive_length)
    parser.add_argument("--groups", nargs="+", choices=["cold", "l1", "l2"])
    parser.add_argument("--repeats", type=driver.positive_length)
    parser.add_argument("--context-length", type=driver.positive_length)
    parser.add_argument("--max-total-tokens", type=driver.positive_length)
    parser.add_argument("--chunked-prefill-size", type=driver.positive_length)
    parser.add_argument("--mem-fraction", type=float)
    parser.add_argument("--hicache-size", type=float, help="每 rank 的主机缓存 GB")
    parser.add_argument("--no-plot", action="store_true", default=None)
    args = parser.parse_args()
    if args.list_experiments:
        print("\n".join(p.stem for p in sorted((PROJECT / "configs/experiments").glob("*.yaml"))))
        return 0
    if not args.experiment:
        parser.error("需要实验配置名")
    job = Job(args)
    if args.dry_run:
        print(json.dumps(dict(experiment=job.experiment, runtime=args.runtime, expected_image=job.lock["image"],
                              source_base=job.lock["sglang_base"], patch_sha256=job.lock["patch_sha256"],
                              model_directory=str(job.model), result_directory=str(job.root),
                              server=job.resolution["server"], test_args=job.test_argv), ensure_ascii=False, indent=2))
        return 0
    return job.execute()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"任务配置错误：{exc}", file=sys.stderr)
        sys.exit(1)
