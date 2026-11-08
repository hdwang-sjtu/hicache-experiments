"""总控使用的源码/模型准备工具；下载阶段与推理计时完全分开。"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import struct
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


def model_storage_root(directory):
    """识别快照所属模型仓库，用于版本检查和缺失文件补齐。"""
    directory = Path(directory).expanduser().resolve()
    if directory.parent.name == "snapshots" and directory.parent.parent.name.startswith("models--"):
        return directory.parent.parent
    return directory


def docker_model_mount(directory):
    directory = Path(directory).expanduser().resolve()
    storage = model_storage_root(directory)
    if storage != directory:
        # HF 新布局可从仓库 blobs 再链接到缓存根的共享 blobs。
        # 挂载缓存根保留整条相对链接；普通模型目录仍只挂载自身。
        storage = storage.parent
    return str(storage), str(Path("/model") / directory.relative_to(storage))


def model_inventory(directory):
    """检查 HF 模型的配置、tokenizer、权重及全部分片，不能只看 config.json。"""
    directory = Path(directory).resolve()
    storage = model_storage_root(directory)
    weight_roots = [directory]
    if storage != directory:
        # 允许所属仓库的旧 blobs 和同一 HF 缓存的共享 blobs。
        # 检查最终目标，避免链接借共享目录再次跳到无关外部路径。
        weight_roots += [storage / "blobs", storage.parent / "blobs"]
    config = directory / "config.json"
    if not config.is_file():
        raise FileNotFoundError("模型缺少 config.json")
    if not isinstance(json.loads(config.read_text()), dict):
        raise ValueError("模型 config.json 无效")
    tokenizers = ["tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt", "vocab.json"]
    if not any((directory / name).is_file() and (directory / name).stat().st_size for name in tokenizers):
        if any((directory / name).is_file() for name in tokenizers):
            raise ValueError("模型 tokenizer 文件为空")
        raise FileNotFoundError("模型缺少 tokenizer 文件")
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
        raise FileNotFoundError("模型没有 safetensors/bin 权重或分片索引")
    for name in weights:
        path = (directory / name).resolve()
        if not any(path.is_relative_to(root) for root in weight_roots):
            raise ValueError(f"模型权重分片位于目录外：{name}")
        if not path.is_file():
            raise FileNotFoundError(f"模型缺少权重分片：{name}")
        if path.stat().st_size <= 0:
            raise ValueError(f"模型权重分片为空：{name}")
    return {str(path.relative_to(directory)): dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and ".cache" not in path.relative_to(directory).parts
            and path.name not in (".hicache-model.json", ".hicache-model.json.tmp")}


def inspect_model(path):
    """索引/大小 + safetensors header 边界检查，不声称已做全权重哈希。"""
    path = Path(path).resolve()
    inventory = model_inventory(path)
    weights = sorted(path.glob('*.safetensors'))
    if not weights:
        raise ValueError('本入口需要 safetensors 权重')
    index = path / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map'] if index.exists() else None
    tensor_names = set()
    for weight in weights:
        with weight.open('rb') as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError(f'无效 safetensors header: {weight.name}')
            size = struct.unpack('<Q', prefix)[0]
            if not 0 < size < min(weight.stat().st_size, 100_000_000):
                raise ValueError(f'无效 safetensors header 长度: {weight.name}')
            header = json.loads(f.read(size))
        end = weight.stat().st_size - 8 - size
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            start, stop = tensor['data_offsets']
            if not 0 <= start < stop <= end or name in tensor_names:
                raise ValueError(f'损坏或重复 tensor: {name}')
            if mapping and mapping.get(name) != weight.name:
                raise ValueError(f'tensor 与分片索引不一致: {name}')
            tensor_names.add(name)
    if mapping and tensor_names != set(mapping):
        raise ValueError('分片中缺少索引指定的 tensor')
    revisions = set()
    metadata = path / '.cache/huggingface/download'
    for item in metadata.glob('*.metadata'):
        first = next(iter(item.read_text().splitlines()), '')
        if re.fullmatch('[0-9a-f]{40}', first):
            revisions.add(first)
    config = json.loads((path / 'config.json').read_text())
    text_config = config.get('text_config', config)
    snapshot_revision = path.name if model_storage_root(path) != path and re.fullmatch('[0-9a-f]{40}', path.name) else None
    return dict(path=str(path), files=inventory, weight_bytes=sum(w.stat().st_size for w in weights),
                revision=snapshot_revision or (next(iter(revisions)) if len(revisions) == 1 else None),
                revision_evidence=('HF snapshot path' if snapshot_revision else 'local HF metadata') + ' (not full content hash)',
                config_sha256=hashlib.sha256((path / 'config.json').read_bytes()).hexdigest(),
                model_type=text_config.get('model_type'), state_chunk_size=text_config.get('mamba_chunk_size', 64))


def prepare_model(repo, revision, directory=None, downloader=None):
    """指定目录则检查/补齐；否则直接使用 HF 返回的默认缓存路径。"""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("模型 revision 必须是固定的完整 commit SHA")
    options = dict(repo_id=repo, revision=revision)
    if directory is not None:
        directory = Path(directory).expanduser().resolve()
        # 只读取现有来源线索，不新建完成标记，也不把线索视为内容哈希。
        marker = directory / ".hicache-model.json"
        if marker.is_file():
            saved = json.loads(marker.read_text())
            if saved.get("repo") != repo or saved.get("revision") != revision:
                raise ValueError("模型目录已有记录与所选模型/版本冲突")
        storage = model_storage_root(directory)
        if storage != directory and (storage.name != "models--" + repo.replace("/", "--") or directory.name != revision):
            raise ValueError("HF 快照目录与所选模型/版本冲突")
        for item in (directory / ".cache/huggingface/download").rglob("*.metadata"):
            recorded = next(iter(item.read_text().splitlines()), "")
            if re.fullmatch(r"[0-9a-f]{40}", recorded) and recorded != revision:
                raise ValueError("模型目录的 HF 下载版本与实验配置冲突")
        try:
            info = inspect_model(directory)
        except FileNotFoundError:
            pass
        else:
            print(f"使用已有模型：{directory}", flush=True)
            return dict(info, repo=repo, requested_revision=revision, source="directory", download_requested=False)
        if storage != directory:
            options["cache_dir"] = str(storage.parent)
        else:
            options["local_dir"] = str(directory)
            # 只补缺失文件，避免覆盖已有权重或让下载库扫描无元数据的大文件。
            options["ignore_patterns"] = [str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()]
    if downloader is None:
        from huggingface_hub import constants, snapshot_download
        downloader = snapshot_download
        if constants.HF_HUB_OFFLINE:
            options["local_files_only"] = True
    print(f"下载/补齐模型 {repo}@{revision} → {directory or 'HF 默认缓存'}", flush=True)
    downloaded = Path(downloader(**options)).resolve()
    if directory is not None and downloaded != directory:
        raise RuntimeError(f"HF 返回路径与指定模型目录不符：{downloaded}")
    info = inspect_model(downloaded)
    return dict(info, repo=repo, requested_revision=revision,
                source="directory" if directory is not None else "huggingface", download_requested=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    source = sub.add_parser("source")
    source.add_argument("--cache", type=Path, required=True)
    source.add_argument("--lock", type=Path, required=True)
    source.add_argument("--patch", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)
    model = sub.add_parser("model")
    model.add_argument("--directory", type=Path, help="完整模型目录；省略时使用 HF 默认缓存")
    model.add_argument("--repo", required=True)
    model.add_argument("--revision", required=True)
    model.add_argument("--output", type=Path, required=True)
    local = sub.add_parser("inspect-model")
    local.add_argument("--directory", type=Path, required=True)
    local.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "source":
        result = prepare_source(args.cache, json.loads(args.lock.read_text()), args.patch)
    elif args.action == "inspect-model":
        result = dict(inspect_model(args.directory), source="local-readonly", reused=True)
    else:
        result = prepare_model(args.repo, args.revision, args.directory)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
