"""总控使用的源码/模型准备工具；下载阶段与推理计时完全分开。"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

DIFF_ARGS = ["diff", "--binary", "--full-index", "--no-ext-diff", "--no-textconv",
             "--src-prefix=a/", "--dst-prefix=b/", "--unified=3", "--diff-algorithm=myers"]


def write_json(path, value):
    """同目录原子替换，状态读取者不会看到半截 JSON。"""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


@contextmanager
def cache_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"缓存正在被另一任务准备：{path}") from exc
        yield


def git(source, *args):
    return subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "-C", str(source), *args],
        stderr=None, timeout=600,
    )


def verify_source(checkout, lock):
    base = git(checkout, "rev-parse", "HEAD").decode().strip()
    patch_hash = hashlib.sha256(git(checkout, *DIFF_ARGS, "HEAD")).hexdigest()
    untracked = git(checkout, "ls-files", "--others", "--exclude-standard").decode().splitlines()
    staged = git(checkout, "diff", "--cached", "--name-only").strip()
    if base != lock["sglang_base"] or patch_hash != lock["patch_sha256"] or untracked or staged:
        raise RuntimeError(f"源码缓存与版本锁不一致，拒绝覆盖或继续使用：{checkout}")


def prepare_source(cache, lock, patch):
    patch = Path(patch).resolve()
    if hashlib.sha256(patch.read_bytes()).hexdigest() != lock["patch_sha256"]:
        raise ValueError("SGLang 补丁 SHA256 与版本锁不符")
    if not re.fullmatch(r"[0-9a-f]{40}", lock["sglang_base"]):
        raise ValueError("SGLang 基线必须是固定的完整 commit SHA")
    cache = Path(cache)
    key = lock["sglang_base"] + "-" + lock["patch_sha256"]
    target = cache / key
    with cache_lock(cache / (key + ".lock")):
        if target.exists():
            verify_source(target / "checkout", lock)
            print(f"复用已校验的源码：{target / 'checkout'}", flush=True)
        else:
            temp = Path(tempfile.mkdtemp(prefix=".preparing-", dir=cache))
            try:
                checkout = temp / "checkout"
                checkout.mkdir()
                git(checkout, "init", "--quiet")
                git(checkout, "remote", "add", "origin", lock["sglang_repository"])
                print(f"获取 SGLang 固定基线 {lock['sglang_base']}", flush=True)
                git(checkout, "fetch", "--depth=1", "--no-tags", "origin", lock["sglang_base"])
                git(checkout, "-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", "FETCH_HEAD")
                git(checkout, "apply", "--check", str(patch))
                git(checkout, "apply", str(patch))
                verify_source(checkout, lock)
                write_json(temp / "manifest.json", lock)
                temp.rename(target)
            finally:
                if temp.exists():
                    shutil.rmtree(temp)  # 仅清理本函数创建的临时目录。
    return dict(path=str((target / "checkout").resolve()), base=lock["sglang_base"],
                patch_sha256=lock["patch_sha256"], repository=lock["sglang_repository"])


def model_inventory(directory):
    """检查 HF 模型的配置、tokenizer、权重及全部分片，不能只看 config.json。"""
    directory = Path(directory).resolve()
    config = directory / "config.json"
    if not config.is_file() or not isinstance(json.loads(config.read_text()), dict):
        raise ValueError("模型缺少有效 config.json")
    tokenizers = ["tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt", "vocab.json"]
    if not any((directory / name).is_file() and (directory / name).stat().st_size for name in tokenizers):
        raise ValueError("模型缺少 tokenizer 文件")
    indexes = [directory / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
               if (directory / name).is_file()]
    weights = set()
    if indexes:
        for index in indexes:
            mapping = json.loads(index.read_text()).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"权重分片索引无效：{index}")
            weights.update(mapping.values())
    else:
        weights.update(name for name in ("model.safetensors", "pytorch_model.bin") if (directory / name).is_file())
    if not weights:
        raise ValueError("模型没有 safetensors/bin 权重或分片索引")
    for name in weights:
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"模型缺少有效权重分片：{name}")
    return {str(path.relative_to(directory)): dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and ".cache" not in path.relative_to(directory).parts
            and path.name not in (".hicache-model.json", ".hicache-model.json.tmp")}


def prepare_model(directory, repo, revision, downloader=None):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("模型 revision 必须是固定的完整 commit SHA")
    directory = Path(directory).resolve()
    marker = directory / ".hicache-model.json"
    with cache_lock(directory.parent / (directory.name + ".lock")):
        if marker.is_file():
            saved = json.loads(marker.read_text())
            if saved.get("repo") != repo or saved.get("revision") != revision:
                raise RuntimeError("模型缓存的仓库/版本与任务不符，拒绝覆盖")
            try:
                complete = saved.get("files") == model_inventory(directory)
            except (OSError, ValueError):
                complete = False
            if complete:
                print(f"复用固定版本模型：{directory}", flush=True)
                return dict(path=str(directory), repo=repo, revision=revision, reused=True)
            # 有完成标记但文件变动，不能依赖 HF 的旧下载元数据误认缓存有效。
            force_download = True
        else:
            force_download = False
        directory.mkdir(parents=True, exist_ok=True)
        if downloader is None:
            # 只影响准备子进程；之后 serve.sh 仍按离线配置启动。
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            from huggingface_hub import snapshot_download
            downloader = snapshot_download
        print(f"下载/补齐模型 {repo}@{revision} → {directory}", flush=True)
        downloader(repo_id=repo, revision=revision, local_dir=str(directory), force_download=force_download)
        files = model_inventory(directory)
        write_json(marker, dict(repo=repo, revision=revision, files=files))
        return dict(path=str(directory), repo=repo, revision=revision, reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    source = sub.add_parser("source")
    source.add_argument("--cache", type=Path, required=True)
    source.add_argument("--lock", type=Path, required=True)
    source.add_argument("--patch", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)
    model = sub.add_parser("model")
    model.add_argument("--directory", type=Path, required=True)
    model.add_argument("--repo", required=True)
    model.add_argument("--revision", required=True)
    model.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "source":
        result = prepare_source(args.cache, json.loads(args.lock.read_text()), args.patch)
    else:
        result = prepare_model(args.directory, args.repo, args.revision)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
