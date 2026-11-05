"""命名服务配置：只解析数据和路径；SGLang 负责实际服务参数的解析。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

from prepare_runtime import docker_model_mount

CONFIG_DIR = Path(__file__).resolve().parent / "configs/serve"
# 兼容原来的显式环境变量；没有设置的变量不产生默认值，不覆盖 YAML。
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
    if not isinstance(data, dict) or set(data) - {"model", "env", "sglang"}:
        raise ValueError("配置顶层只支持 model、env、sglang")
    model = data.get("model")
    if (not isinstance(model, str) or not model or Path(model).is_absolute()
            or ".." in Path(model).parts or "\0" in model):
        raise ValueError("model 必须是 MODEL_ROOT 下的相对模型目录名")
    if not isinstance(data.get("sglang"), dict) or not isinstance(data.get("env", {}), dict):
        raise ValueError("sglang 和 env 必须是 YAML 映射")
    # model-path 由顶层 model 解析，避免文件出现两个互相矛盾的模型来源。
    for key in data["sglang"]:
        if not re.fullmatch(r"[a-z][a-z0-9-]*", key) or key in ("model-path", "model", "config"):
            raise ValueError(f"sglang 参数请使用原生连字符名称，模型只写顶层 model：{key}")
    return path, raw, data


def resolve(name, runtime, argv, environ=None, directory=CONFIG_DIR, resolve_model=True):
    """优先级：CLI > 显式环境变量 > 文件；模型 CLI 覆盖需先处理挂载路径。"""
    environ = os.environ if environ is None else environ
    path, raw, data = read_profile(name, directory)
    server = dict(data["sglang"])
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
            if not environ.get("MODEL_ROOT"):
                raise ValueError("请设置 MODEL_ROOT；也可用 MODEL_DIR 或 --model-path 指定完整模型目录")
            model_dir = str(Path(environ["MODEL_ROOT"]).expanduser() / data["model"])
        model_dir = str(Path(model_dir).expanduser().resolve())
        server["model-path"] = docker_model_mount(model_dir)[1] if runtime == "docker" else model_dir

    environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for key, value in data.get("env", {}).items():
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
    if any("\0" in value for value in [model_dir or "", *environment.values(), *cli]):
        raise ValueError("路径、参数和环境变量不能包含 NUL 字符")
    yaml = yaml_module()
    return dict(
        name=name, source_path=str(path.resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
        source_text=raw.decode("utf-8"), runtime=runtime, model_directory=model_dir,
        environment=environment, server=server, cli_args=cli,
        server_yaml=yaml.safe_dump(server, allow_unicode=True, sort_keys=False),
    )


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
