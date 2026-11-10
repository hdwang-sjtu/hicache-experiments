"""命名服务配置：只解析数据和路径；SGLang 负责实际服务参数的解析。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

from hicache.prepare import docker_model_mount

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/experiments"
# 环境变量只做显式覆盖；服务默认值只有 YAML 一处来源。
ENV_OPTIONS = {
    "HOST": ("host", str), "PORT": ("port", int), "TP": ("tp-size", int), "EP": ("ep-size", int),
    "CHUNKED_PREFILL_SIZE": ("chunked-prefill-size", int),
    "DTYPE": ("dtype", str), "KV_DTYPE": ("kv-cache-dtype", str),
    "CONTEXT_LENGTH": ("context-length", int), "MAX_TOTAL_TOKENS": ("max-total-tokens", int),
    "MEM_FRACTION": ("mem-fraction-static", float), "HICACHE_SIZE": ("hicache-size", float),
}


def boolean(value, name):
    """环境变量中的布尔开关可写 0/1 或 false/true。"""
    if str(value).lower() not in ("0", "1", "false", "true"):
        raise ValueError(f"{name} 必须为 0/1 或 false/true")
    return str(value).lower() in ("1", "true")


def yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("命名配置需要 PyYAML：python3 -m pip install -r requirements-launch.txt") from exc
    return yaml


def read_profile(name, directory):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("配置名只能包含字母、数字、点、下划线和连字符，不接受路径")
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"找不到配置 {name!r}；用 --list-configs 查看可用名称")
    raw = path.read_bytes()
    yaml = yaml_module()

    # YAML 默认会静默接受重复键；在这里拒绝，避免用户以为前一项仍然有效。
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        values = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in values:
                raise ValueError(f"配置键必须为不重复的字符串：{key!r}")
            values[key] = loader.construct_object(value_node)
        return values

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    data = yaml.load(raw.decode("utf-8"), Loader=UniqueLoader)
    if not isinstance(data, dict) or set(data) != {"model", "server", "test"}:
        raise ValueError("实验配置必须包含 model、server、test")
    model = data.get("model")
    if (not isinstance(model, dict) or set(model) != {"repo", "revision"}
            or not isinstance(model["repo"], str) or not model["repo"]
            or not re.fullmatch(r"[0-9a-f]{40}", str(model["revision"]))):
        raise ValueError("model 需要 repo 和固定的完整 revision SHA")
    settings = data["server"]
    if (not isinstance(settings, dict) or set(settings) - {"load-format", "routing", "prefill-graph", "env", "sglang"}
            or not isinstance(settings.get("sglang"), dict) or not isinstance(settings.get("env", {}), dict)
            or not isinstance(data["test"], dict)):
        raise ValueError("server.sglang、server.env 和 test 必须是 YAML 映射")
    # model-path 由顶层 model 解析，避免文件出现两个互相矛盾的模型来源。
    for key in settings["sglang"]:
        if not re.fullmatch(r"[a-z][a-z0-9-]*", key) or key in ("model-path", "model", "config", "load-format", "cuda-graph-backend-prefill", "cuda-graph-bs-prefill"):
            raise ValueError(f"sglang 参数请使用原生连字符名称，模型只写顶层 model：{key}")
    return path, raw, data


def resolve(name, runtime, argv, environ=None, directory=CONFIG_DIR, resolve_model=True, data=None):
    """优先级：CLI > 显式环境变量 > 文件；模型 CLI 覆盖需先处理挂载路径。"""
    environ = os.environ if environ is None else environ
    path, raw, original = read_profile(name, directory)
    data = original if data is None else data
    settings = data["server"]
    server = dict(settings["sglang"])
    for variable, (option, convert) in ENV_OPTIONS.items():
        if variable in environ:
            server[option] = convert(environ[variable])
    if "HICACHE" in environ:
        enabled = boolean(environ["HICACHE"], "HICACHE")
        server.update({"enable-hierarchical-cache": enabled, "enable-hicache-experiment": enabled})

    model_override = None
    cli = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--config" or arg.startswith("--config="):
            raise ValueError("命名配置不能再叠加 --config；请编辑配置或追加单项参数")
        if arg in ("--model-path", "--model"):
            index += 1
            if index == len(argv) or not argv[index] or argv[index].startswith("--"):
                raise ValueError("--model-path 缺少路径")
            model_override = argv[index]
        elif arg.startswith(("--model-path=", "--model=")):
            model_override = arg.split("=", 1)[1]
            if not model_override:
                raise ValueError("--model-path 缺少路径")
        else:
            cli.append(arg)
        index += 1
    model_dir = None
    if resolve_model:
        model_dir = model_override or environ.get("MODEL_DIR")
        if not model_dir:
            raise ValueError("请设置 MODEL_DIR 或 --model-path 指向已准备的模型/配置目录")
        model_dir = str(Path(model_dir).expanduser().resolve())
        server["model-path"] = docker_model_mount(model_dir)[1] if runtime == "docker" else model_dir

    environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for key, value in settings.get("env", {}).items():
        if not (re.fullmatch(r"SGLANG_[A-Z0-9_]+", key) or key in environment):
            raise ValueError(f"env 只保存 SGLANG_* 和离线开关；部署变量请在启动环境设置：{key}")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"环境变量 {key} 必须为标量")
        environment[key] = str(int(value)) if isinstance(value, bool) else str(value)
    # 即使文件未写出两个实验开关，显式运行环境也能覆盖它们。
    for key in set(environment) | {"SGLANG_ENABLE_METRICS_DEVICE_TIMER", "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"}:
        if key in environ:
            environment[key] = environ[key]
    if "FORWARD_TIMING" in environ:
        environment["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] = str(int(boolean(environ["FORWARD_TIMING"], "FORWARD_TIMING")))
    enabled = server.get("enable-hierarchical-cache", False)
    if not isinstance(enabled, bool):
        raise ValueError("enable-hierarchical-cache 请使用 YAML 的 true/false")
    # CLI 只接受 SGLang 原生语法，store_true 标志不存在 '--flag false' 写法。
    enabled = enabled or "--enable-hierarchical-cache" in cli
    timer = boolean(environment.get("SGLANG_ENABLE_METRICS_DEVICE_TIMER", "0"), "SGLANG_ENABLE_METRICS_DEVICE_TIMER")
    if timer and not enabled:
        raise ValueError("GPU 计时需要开启 HiCache；关闭 HiCache 时也请设置 FORWARD_TIMING=0")
    if enabled:
        environment.setdefault("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND", "python")
    configure_modes(settings, server, environment)
    # 独立服务入口用 YAML 的测试网格；总控会在容量覆盖后重新生成。
    if resolve_model:
        configure_graph(server, data["test"])
    if any("\0" in value for value in [model_dir or "", *environment.values(), *cli]):
        raise ValueError("路径、参数和环境变量不能包含 NUL 字符")
    yaml = yaml_module()
    return dict(
        name=name, source_path=str(path.resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
        source_text=raw.decode("utf-8"), runtime=runtime, model_directory=model_dir,
        environment=environment, server=server, cli_args=cli,
        server_yaml=yaml.safe_dump(server, allow_unicode=True, sort_keys=False),
    )


def positive_length(value):
    """1K=1024；配置和两个 Python 入口共用同一种长度语法。"""
    try:
        value = str(value)
        number = int(value[:-1]) * 1024 if value.lower().endswith("k") else int(value)
        if number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Expected a positive integer or e.g. 8k (8192)")


def configure_modes(settings, server, environment):
    """模式只有一个来源；明确清零另一路由开关，避免继承环境造成冲突。"""
    for key, choices, default in (
        ("load-format", ("dummy", "auto"), "dummy"),
        ("routing", ("round-robin", "original"), "round-robin"),
        ("prefill-graph", ("breakable", "disabled"), "breakable"),
    ):
        value = settings.get(key, default)
        if value not in choices:
            raise ValueError(f"server.{key} 只支持 {choices}")
        settings[key] = value
    server["load-format"] = settings["load-format"]
    server["cuda-graph-backend-prefill"] = settings["prefill-graph"]
    server["cuda-graph-backend-decode"] = "disabled"
    environment["SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS"] = str(int(settings["routing"] == "round-robin"))
    environment["SGLANG_SIMULATE_UNIFORM_EXPERTS"] = "0"


def forward_chunks(tokens, chunk):
    """返回逻辑 forward 长度；不把输出的首 token 当作一次 decode。"""
    return [min(chunk, tokens - start) for start in range(0, tokens, chunk)]


def configure_graph(server, test):
    """捕获正式请求所需的精确 token 桶；reference/prepare 允许 eager。"""
    if server["cuda-graph-backend-prefill"] == "disabled":
        server.pop("cuda-graph-bs-prefill", None)
        return
    chunk = server.get("chunked-prefill-size")
    if not isinstance(chunk, int) or chunk <= 0:
        raise ValueError("Graph 实验需要固定正数 chunked-prefill-size")
    prefixes = [positive_length(v) for v in test["prefix-lengths"]]
    suffixes = [positive_length(v) for v in test["suffix-lengths"]]
    buckets = set()
    for p in prefixes:
        for s in suffixes:
            for group in test["groups"]:
                if group != "cold" and s > chunk:
                    raise ValueError("L1/L2 的 S 不能超过 chunk")
                buckets.update(forward_chunks(p + s if group == "cold" else s, chunk))
    if not buckets:
        raise ValueError("Graph 实验网格不能为空")
    server["cuda-graph-bs-prefill"] = sorted(buckets)
    server["cuda-graph-max-seq-len-prefill"] = server["context-length"]


def build_run_parser():
    """返回 run 参数解析器及公开 Action 映射，CLI/YAML 共用类型和默认值。"""
    p = argparse.ArgumentParser(add_help=False)
    actions = {}

    def add(option, **kwargs):
        actions[option.removeprefix("--")] = p.add_argument(option, **kwargs)

    add(
        "--base-url", default="http://127.0.0.1:39081", help="SGLang HTTP 服务地址"
    )
    add("--tokenizer", required=True, help="与服务端模型匹配的本地 tokenizer 目录")
    add("--output-dir", required=True, help="本次结果目录，不可覆盖已有测量")
    add("--server-artifacts", help="可选：客户端可读的服务启动记录目录或其副本（含 launch.json）")
    add(
        "--prefix-lengths", nargs="+", type=positive_length,
        default=[8192, 16384, 32768], help="历史 P 的 token 数，可写 8k 16k 32k",
    )
    add(
        "--suffix-lengths", nargs="+", type=positive_length, default=[128],
        help="新增 S 的 token 数，可一次指定多个值",
    )
    add(
        "--output-length", type=positive_length, default=1, help="每个测量请求生成的 token 数"
    )
    add(
        "--groups", nargs="+", choices=["cold", "l1", "l2"], default=["cold", "l1"],
        help="测量哪些缓存组；L2 需要开启 HiCache 和实验接口",
    )
    add(
        "--repeats", type=positive_length, default=20, help="每个 P/S/缓存组所需的有效样本数"
    )
    add(
        "--max-attempts", type=positive_length,
        help="每组最大尝试次数，默认 repeats 的 3 倍；无效样本也消耗次数",
    )
    add(
        "--warmup-rounds", type=positive_length, default=1,
        help="每个 P/S 组合的运行预热轮数，不计入有效样本",
    )
    add("--seed", type=int, default=0, help="控制各组执行顺序的随机种子")
    add("--text-file", help="固定 UTF-8 前缀文本；省略时使用内置英文笔记")
    add("--timeout", type=positive_length, default=600, help="单次 HTTP 请求超时，单位秒")
    add("--ready-timeout", type=positive_length, default=600, help="等待服务就绪的上限，单位秒")
    add("--no-plot", action="store_true", help="保存 CSV 汇总但不绘图")
    add("--profile", action="store_true", help="只在测量请求期间采集 CPU/GPU trace；结果含 Profiler 开销，仅用于诊断")
    add("--server-profile-dir", help="Profiler 写入的服务端绝对路径；启用 --profile 时必须指定")
    add("--gpu-timing", action="store_true", help="记录原生 extend CUDA Event 区间；服务需 FORWARD_TIMING=1，区间可能包含等待")
    add("--prefill-graph", choices=('breakable', 'disabled'), help="要求正式请求在所有 rank 使用指定路径")
    add("--note", default="", help="保存在配置和图中的实验说明，例如未缩放 RoPE 外推；不影响请求内容")
    add("--telemetry-dir", help="宿主机采样目录；启用后关联请求窗口，监控缺口或额外 GPU 进程不计入正式有效数")
    add("--telemetry-same-host", action="store_true", help="确认客户端与遥测进程在同一主机且共用单调时钟；远程 SSH 隧道不满足此条件")

    return p, actions


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


def unpack(resolution):
    """给 Bash 输出 NUL 分隔的数组，避免 eval 或执行 YAML 中的字符串。"""
    environment = [f"{key}={value}" for key, value in resolution["environment"].items()]
    values = [resolution["model_directory"], resolution["server"]["model-path"],
              resolution["server_yaml"], str(len(environment)), *environment, *resolution["cli_args"]]
    sys.stdout.buffer.write(b"".join(value.encode() + b"\0" for value in values))


def preview(resolution):
    print(f"配置：{resolution['name']}\n模型目录：{resolution['model_directory']}")
    print("下发给 SGLang 的 YAML（命令行仍可覆盖）：")
    print(resolution["server_yaml"], end="")
    print("运行时环境：" + shlex.join([f"{k}={v}" for k, v in resolution["environment"].items()]))
    print("命令行覆盖：" + shlex.join(resolution["cli_args"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    p = sub.add_parser("resolve")
    p.add_argument("--runtime", choices=["native", "docker"], required=True)
    p.add_argument("name")
    p.add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("unpack")
    sub.add_parser("preview")
    p = sub.add_parser("model-mount")
    p.add_argument("directory")
    args = parser.parse_args()
    if args.action == "list":
        print("\n".join(path.stem for path in sorted(CONFIG_DIR.glob("*.yaml"))))
    elif args.action == "resolve":
        argv = args.args[1:] if args.args[:1] == ["--"] else args.args
        print(json.dumps(resolve(args.name, args.runtime, argv), ensure_ascii=False))
    elif args.action == "unpack":
        unpack(json.load(sys.stdin))
    elif args.action == "model-mount":
        sys.stdout.buffer.write(b"".join(value.encode() + b"\0" for value in docker_model_mount(args.directory)))
    else:
        preview(json.load(sys.stdin))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        sys.exit(2)
