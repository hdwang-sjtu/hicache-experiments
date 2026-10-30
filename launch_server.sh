#!/usr/bin/env bash
# 在宿主机运行本脚本：创建容器 → 启动 SGLang → 持续显示服务日志。
# 保持此终端打开，在另一个终端执行测试命令；按 Ctrl-C 停止本次容器。
set -euo pipefail

# 1. 实验参数：使用环境变量覆盖默认值，例如 GPU=2 RUN_ID=test-01 bash ...。
#    宿主机路径在这里配置；容器中的路径固定为 /model、/experiment、/results。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${SGLANG_DIR:-$SCRIPT_DIR/../sglang}"
MODEL_DIR="${MODEL_DIR:-/state/partition/hdwang/models/Qwen3-8B}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONTAINER_NAME="${CONTAINER_NAME:-hicache-exp-$(id -u)-$RUN_ID}"
IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
GPU="${GPU:-1}"
PORT="${PORT:-39081}"

# --help 和 --dry-run 都不会启动容器，适合先检查用法与最终命令。
if [[ ${1:-} == --help ]]; then
    cat <<'EOF'
用法：[环境变量=值 ...] bash launch_server.sh [--dry-run] [SGLang 参数 ...]
默认：Qwen3-8B、GPU 1、TP=1、BF16 权重和 KV、关闭 HiCache 和 CUDA Graph。

常用环境变量：
  MODEL_DIR / SGLANG_DIR   宿主机上的模型目录 / SGLang 源码目录
  GPU / PORT              使用的宿主机 GPU 编号 / HTTP 端口
  RUN_ID / CONTAINER_NAME  本次结果目录名 / 容器名
  IMAGE / RESULTS_DIR     本地镜像 / 宿主机结果根目录
  TP / DTYPE / KV_DTYPE    TP 大小 / 权重精度 / KV 精度
  CONTEXT_LENGTH          最大上下文长度
  MAX_TOTAL_TOKENS        GPU KV 池的 token 容量上限
  MEM_FRACTION            SGLang 的静态显存预算比例

示例：
  bash launch_server.sh --dry-run
  GPU=1 RUN_ID=baseline-8k bash launch_server.sh
  MODEL_DIR=/path/to/model DTYPE=float16 bash launch_server.sh

额外参数会追加到 SGLang 启动命令；镜像和模型需提前在本机准备好。
EOF
    exit 0
fi
dry_run=false
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi
[[ ${1:-} != -- ]] || shift

# RUN_ID 只允许单层目录名，避免结果被写到预期目录之外。
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_.-]+$ && "$RUN_ID" != . && "$RUN_ID" != .. ]] || {
    echo 'RUN_ID must be a single directory name.' >&2
    exit 2
}
[[ -f "$MODEL_DIR/config.json" && -d "$SGLANG_DIR/python/sglang" ]] || {
    echo 'Check MODEL_DIR (local model) and SGLANG_DIR (source checkout).' >&2
    exit 2
}
SGLANG_DIR="$(cd "$SGLANG_DIR" && pwd)"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
[[ "$RESULTS_DIR" == /* ]] || RESULTS_DIR="$PWD/$RESULTS_DIR"

# 2. 组装命令。使用 Bash 数组，确保路径含空格时仍是一个完整参数。
#    阶段 1：只测 Cold/L1；单卡、单请求，保留 overlap 和 chunked prefill。
#    --enable-metrics 用于返回现有服务端时间，两个 Graph 开关保持配置一致。
server_args=(python -m sglang.launch_server
    --model-path /model --host 127.0.0.1 --port "$PORT"
    --tp-size "${TP:-1}" --dtype "${DTYPE:-bfloat16}"
    --kv-cache-dtype "${KV_DTYPE:-bfloat16}"
    --context-length "${CONTEXT_LENGTH:-40960}"
    --max-total-tokens "${MAX_TOTAL_TOKENS:-36864}"
    --mem-fraction-static "${MEM_FRACTION:-0.7}"
    --max-running-requests 1 --chunked-prefill-size 4096 --page-size 1
    --enable-metrics --return-output-ids
    --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
    "$@"
)

# 源码、权重和实验脚本只读挂载；只有 /results 可写。
# PYTHONPATH 确保服务使用当前 checkout 的源码，而不是镜像自带的源码。
# --network=host 使宿主机和容器都能通过 127.0.0.1:PORT 访问服务。
docker_args=(docker run --detach --rm --name "$CONTAINER_NAME" --pull never
    --gpus "device=$GPU" "--ipc=host" "--network=host"
    --mount "type=bind,src=$SGLANG_DIR,dst=/sgl-workspace/sglang,readonly"
    --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly"
    --mount "type=bind,src=$SCRIPT_DIR,dst=/experiment,readonly"
    --mount "type=bind,src=$RESULTS_DIR,dst=/results"
    --workdir /sgl-workspace/sglang
    --env "PYTHONPATH=/sgl-workspace/sglang/python"
    --env "SGLANG_SOURCE_DIR=/sgl-workspace/sglang"
    --env "EXPERIMENT_LAUNCH_DIR=/results/$RUN_ID"
    --env "HF_HUB_OFFLINE=1" --env "TRANSFORMERS_OFFLINE=1" --entrypoint /usr/bin/env
)

if "$dry_run"; then
    printf '%q ' "${docker_args[@]}" "$IMAGE" "${server_args[@]}"
    printf '\n'
    echo '仅打印命令。实际启动时会将本地镜像标签解析为不可变的镜像 ID。'
    exit 0
fi

# 3. 保存复现信息。先固定镜像 ID，避免 latest 标签变化影响本次启动。
image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
mkdir -p "$RESULTS_DIR"
run_dir="$RESULTS_DIR/$RUN_ID"
mkdir "$run_dir" # 目录已存在就报错，保护之前的实验记录。

# 下面这段 Python 仅保存 JSON 和 Git 补丁，不负责启动或测试模型。
python3 - "$run_dir" "$SGLANG_DIR" "$MODEL_DIR" "$image_id" "$GPU" \
    "${docker_args[@]}" "$image_id" "${server_args[@]}" <<'PY'
import json
import pathlib
import subprocess
import sys

out, source, model, image, gpu = sys.argv[1:6]


def git(*args):
    """只读取这份 SGLang checkout 的版本和改动，不修改 Git 全局设置。"""
    return subprocess.check_output(
        ['git', '-c', f'safe.directory={source}', '-C', source, *args],
        text=True,
    )


p = pathlib.Path(out)
(p / 'launch.json').write_text(json.dumps(dict(
    sglang_source=source, model_directory=model, image_id=image, gpu=gpu,
    sglang_commit=git('rev-parse', 'HEAD').strip(),
    sglang_status=git('status', '--porcelain'), command=sys.argv[6:],
), indent=2) + '\n')
(p / 'sglang.patch').write_text(git('diff', '--binary', 'HEAD'))
# 未跟踪的 Python 源码不会出现在 git diff 中，需先跟踪再运行实验。
untracked = git('ls-files', '--others', '--exclude-standard').splitlines()
if any(name.startswith('python/') and name.endswith('.py') for name in untracked):
    raise SystemExit('Untracked SGLang Python sources exist; track them before benchmarking.')
PY
# GPU 信息获取失败时仍保留报错文本；真正的 GPU 可用性由服务启动验证。
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used \
    --format=csv > "$run_dir/gpu.txt" 2>&1 || true

# 4. 创建容器并接管其生命周期。只按本次返回的 ID 清理，不操作其他容器。
container_id="$("${docker_args[@]}" "$image_id" "${server_args[@]}")"
cleanup() {
    docker stop --time 15 "$container_id" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "服务日志：$run_dir/server.log"
echo '在另一个宿主机终端执行以下命令，运行 8K 小规模验证：'
# 使用宿主机 UID/GID 运行 Driver，使结果文件属于当前用户；也适用于 NFS。
printf '%q ' docker exec --user "$(id -u):$(id -g)" "$CONTAINER_NAME" \
    python /experiment/run_experiment.py run --base-url "http://127.0.0.1:$PORT" \
    --tokenizer /model --output-dir "/results/$RUN_ID" \
    --prefix-lengths 8k --suffix-lengths 128 --repeats 3
printf '\n'
# 同时在终端显示日志、在宿主机保存日志；Ctrl-C 会触发上面的清理函数。
docker logs --follow "$container_id" 2>&1 | tee "$run_dir/server.log"
