"""两个启动入口共用的记录工具，仅使用标准库，不参与请求或模型计算。"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

from hicache.prepare import write_json

PROJECT = Path(__file__).resolve().parents[1]


def code_files(project=PROJECT):
    """归档整个小工具包；不再分别维护 Driver/启动器的易遗漏模块清单。"""
    project = Path(project)
    # 远程客户端可以只复制 run_experiment.py 和包，不要求拥有 Shell 启动器。
    entries = tuple(name for name in ("run_all.py", "run_experiment.py", "serve.sh", "launch_server.sh")
                    if (project / name).is_file())
    configuration = tuple(str(p.relative_to(project)) for pattern in (
        'configs/experiments/*.yaml', 'configs/runtime.lock.json', 'patches/*.patch')
                          for p in sorted(project.glob(pattern)) if p.is_file())
    return (*entries, *configuration,
            *(str(p.relative_to(project)) for folder in ("hicache", "tools")
              for p in sorted((project / folder).glob("*.py"))))


def snapshot_code(destination, project=PROJECT):
    """保留包目录和相对导入；整个运行目录搬走后仍能运行已冻结的入口。"""
    destination, project = Path(destination), Path(project)
    hashes = {}
    for name in code_files(project):
        content = (project / name).read_bytes()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    return hashes


def record(args):
    directory = Path(args.directory)
    source = Path(args.source_dir).resolve() if args.source_dir else None
    metadata = dict(
        runtime=args.runtime, started_utc=datetime.now(timezone.utc).isoformat(),
        model_directory=str(Path(args.model_dir).resolve()),
        sglang_source=str(source) if source else None,
        image_id=args.image_id, image_ref=args.image_ref, gpu=args.gpu,
        command=args.command[1:] if args.command[:1] == ["--"] else args.command,
    )
    if source and shutil.which("git"):
        def git(*command):
            return subprocess.check_output(
                ["git", "-c", f"safe.directory={source}", "-C", str(source), *command],
                text=True, stderr=subprocess.PIPE,
            )
        try:
            root = Path(git("rev-parse", "--show-toplevel").strip()).resolve()
        except subprocess.CalledProcessError:
            root = None
        if root == source:
            # 未跟踪的源码不在 git diff 中，不能静默漏掉。
            untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
            if any(name.startswith("python/") and name.endswith(".py") for name in untracked):
                raise ValueError("存在未跟踪的 SGLang Python 源码，请先纳入 Git。")
            metadata.update(sglang_commit=git("rev-parse", "HEAD").strip(),
                            sglang_status=git("status", "--porcelain"))
            (directory / "sglang.patch").write_text(git("diff", "--binary", "HEAD"))

    # 宿主机包版本不能冒充 Docker 内的版本；native 才记录当前 Python 环境。
    if args.runtime == "native":
        spec = importlib.util.find_spec("sglang")
        if spec is None:
            raise ValueError("当前 Python 环境找不到 SGLang；检查 PYTHON_BIN / SGLANG_DIR。")
        metadata["sglang_module"] = spec.origin
        metadata["packages"] = {}
        for package in ("sglang", "torch", "transformers"):
            try:
                metadata["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        metadata["environment"] = {
            key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "SGLANG_ENABLE_METRICS_DEVICE_TIMER",
                "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS", "SGLANG_SIMULATE_UNIFORM_EXPERTS",
            )
        }
    metadata["source_version_available"] = "sglang_commit" in metadata
    if args.profile_json:
        profile = json.loads(args.profile_json)
        (directory / "profile.yaml").write_bytes(profile["source_text"].encode("utf-8"))
        (directory / "server-config.yaml").write_text(profile["server_yaml"])
        (directory / "profile-resolution.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
        metadata["profile"] = {
            key: profile[key] for key in ("name", "source_path", "source_sha256", "environment", "cli_args")
        }
        metadata["profile"]["server_config_sha256"] = hashlib.sha256(profile["server_yaml"].encode()).hexdigest()
    # 总控已把所有脚本冻结在同一结果包的 code/；独立启动才复制 launch-source/。
    # 路径相对 launch.json 保存，整个结果目录换机器后仍可追溯到源码。
    managed = PROJECT == directory.resolve().parent / "code"
    snapshot = directory.parent / "code" if managed else directory / "launch-source"
    if not managed:
        snapshot.mkdir()
    metadata["launcher_source"] = os.path.relpath(snapshot, directory)
    metadata["launcher_sha256"] = (
        {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in code_files()}
        if managed else snapshot_code(snapshot))
    write_json(directory / "launch.json", metadata)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used", "--format=csv"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10,
        )
        gpu_info = result.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_info = str(exc) + "\n"
    (directory / "gpu.txt").write_text(gpu_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--runtime", choices=["native", "docker"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-dir", default="")
    parser.add_argument("--gpu", default="inherited")
    parser.add_argument("--image-id")
    parser.add_argument("--image-ref")
    parser.add_argument("--profile-json", help="启动器提供的已解析命名配置")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    record(parser.parse_args())
