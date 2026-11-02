#!/usr/bin/env bash
# 宿主机入口：准备 Docker 环境并调用 serve.sh，不重复定义 SGLang 参数。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
用法：bash launch_server.sh [配置名] [--dry-run] [SGLang 参数 ...]
  MODEL_ROOT=/models GPU=0 bash launch_server.sh qwen3-8b-4k
  bash launch_server.sh --list-configs
无配置名时兼容 MODEL_DIR=/path/to/model bash launch_server.sh。
需要宿主机 Docker + NVIDIA runtime、本地已有镜像和模型。
已有 SGLang 环境（包括 Vast 普通容器）请直接用 bash serve.sh。
  MODEL_ROOT / MODEL_DIR   模型根目录 / 完整模型目录覆盖（均为宿主机路径）
  SGLANG_DIR               源码目录；显式 SGLANG_DIR='' 使用镜像安装包
  IMAGE                    默认 lmsysorg/sglang:latest，启动前固定为本地镜像 ID
  GPU                      宿主机 GPU 编号或 UUID，默认 0
  RESULTS_DIR / RUN_ID      宿主机结果根目录 / 本轮启动目录名
  CONTAINER_NAME           本次容器名
HICACHE、FORWARD_TIMING、PORT、DTYPE 等原样传给 serve.sh；详见 bash serve.sh --help。
源码、模型和脚本只读挂载；结果可写。默认监听 127.0.0.1:39081。
Ctrl-C 只停止并删除本脚本创建的容器。--dry-run 不访问 Docker 或 GPU。
HELP
    exit 0
fi
dry_run=false
profile=""
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --list-configs) python3 "$SCRIPT_DIR/serve_config.py" list; exit ;;
        --) shift; break ;;
        -*) break ;;
        *) [[ -z "$profile" ]] || { echo '只能选择一个配置名。' >&2; exit 2; }
           profile="$1"; shift ;;
    esac
done
profile_json=""
profile_environment=()
if [[ -n "$profile" ]]; then
    profile_json="$(python3 "$SCRIPT_DIR/serve_config.py" resolve --runtime docker "$profile" -- "$@")"
    mapfile -d '' -t values < <(python3 "$SCRIPT_DIR/serve_config.py" unpack <<< "$profile_json")
    ((${#values[@]} >= 4)) || exit 2
    MODEL_DIR="${values[0]}"
    env_count="${values[3]}"
    profile_environment=("${values[@]:4:env_count}")
else
    MODEL_DIR="${MODEL_DIR:?请设置 MODEL_DIR 为宿主机模型目录，或选择命名配置并设置 MODEL_ROOT}"
fi
IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
GPU="${GPU:-0}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONTAINER_NAME="${CONTAINER_NAME:-hicache-exp-$(id -u)-$RUN_ID}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_.-]+$ && "$RUN_ID" != . && "$RUN_ID" != .. ]] || {
    echo 'RUN_ID 必须是单层目录名。' >&2; exit 2;
}
if [[ ! -v SGLANG_DIR ]]; then
    SGLANG_DIR=""
    [[ ! -d "$SCRIPT_DIR/../sglang/python/sglang" ]] || SGLANG_DIR="$SCRIPT_DIR/../sglang"
fi
[[ "$MODEL_DIR" == /* ]] || MODEL_DIR="$PWD/$MODEL_DIR"
[[ "$RESULTS_DIR" == /* ]] || RESULTS_DIR="$PWD/$RESULTS_DIR"
if [[ -n "$SGLANG_DIR" && "$SGLANG_DIR" != /* ]]; then SGLANG_DIR="$PWD/$SGLANG_DIR"; fi
# host 网络保留 loopback 控制接口语义；远程客户端通过 SSH 隧道访问。
# 不把物理 GPU 编号再次传为容器 CUDA_VISIBLE_DEVICES，避免二次编号选择。
docker_args=(docker run --detach --name "$CONTAINER_NAME" --pull never
    --cidfile "$RESULTS_DIR/$RUN_ID/container.id"
    --gpus "device=$GPU" --ipc=host --network=host
    --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly"
    --mount "type=bind,src=$SCRIPT_DIR,dst=/experiment,readonly"
    --mount "type=bind,src=$RESULTS_DIR,dst=/results"
    --workdir /experiment --entrypoint /usr/bin/env
    --env MODEL_DIR=/model --env "EXPERIMENT_LAUNCH_DIR=/results/$RUN_ID"
    --env MPLCONFIGDIR=/tmp/matplotlib
)
if [[ -n "$SGLANG_DIR" ]]; then
    docker_args+=(--mount "type=bind,src=$SGLANG_DIR,dst=/sgl-workspace/sglang,readonly"
        --env SGLANG_DIR=/sgl-workspace/sglang)
else
    docker_args+=(--env SGLANG_DIR=)
fi
# 只传用户设置的参数；默认值统一由 serve.sh 决定。
if [[ -n "$profile_json" ]]; then
    # 在宿主机解析一次后冻结；容器不再次读取可变配置或宿主机环境变量。
    for entry in "${profile_environment[@]}"; do docker_args+=(--env "$entry"); done
    command=(bash /experiment/serve.sh --exec --resolved-profile "/results/$RUN_ID/profile-resolution.json")
else
    for key in HOST PORT TP DTYPE KV_DTYPE CONTEXT_LENGTH MAX_TOTAL_TOKENS MEM_FRACTION \
        HICACHE HICACHE_SIZE FORWARD_TIMING HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; do
        [[ ! -v "$key" ]] || docker_args+=(--env "$key=${!key}")
    done
    command=(bash /experiment/serve.sh --exec "$@")
fi
if "$dry_run"; then
    [[ -z "$profile_json" ]] || python3 "$SCRIPT_DIR/serve_config.py" preview <<< "$profile_json"
    printf '%q ' "${docker_args[@]}" "$IMAGE" "${command[@]}"
    printf '\n'
    exit 0
fi
[[ -f "$MODEL_DIR/config.json" ]] || { echo 'MODEL_DIR 中没有 config.json。' >&2; exit 2; }
if [[ -n "$SGLANG_DIR" && ! -d "$SGLANG_DIR/python/sglang" ]]; then
    echo 'SGLANG_DIR 中没有 python/sglang。' >&2; exit 2
fi
image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
mkdir -p "$RESULTS_DIR"
run_dir="$RESULTS_DIR/$RUN_ID"
mkdir "$run_dir"
# 在宿主机写记录，保留 NFS root-squash 下的普通用户写入能力。
metadata_args=()
[[ -z "$profile_json" ]] || metadata_args+=(--profile-json "$profile_json")
python3 "$SCRIPT_DIR/launch_metadata.py" --directory "$run_dir" --runtime docker \
    --model-dir "$MODEL_DIR" --source-dir "$SGLANG_DIR" --gpu "$GPU" \
    --image-id "$image_id" --image-ref "$IMAGE" "${metadata_args[@]}" \
    -- "${docker_args[@]}" "$image_id" "${command[@]}"
container_id=""
cleanup() {
    if [[ -z "$container_id" && -s "$run_dir/container.id" ]]; then
        container_id="$(cat "$run_dir/container.id")"
    fi
    [[ -n "$container_id" ]] || return 0
    docker stop --time 15 "$container_id" >/dev/null 2>&1 || true
    docker rm "$container_id" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
container_id="$("${docker_args[@]}" "$image_id" "${command[@]}")"
# --cidfile 在创建时记录身份，启动失败时总控仍能按确切 ID 清理。
[[ -s "$run_dir/container.id" ]] || printf '%s\n' "$container_id" > "$run_dir/container.id"
echo "启动记录：$run_dir；容器：$CONTAINER_NAME"
echo '服务地址见启动配置；客户端通过 --base-url 指定地址，也可在容器内运行 Driver。'
docker logs --follow "$container_id" 2>&1 | tee "$run_dir/server.log"
# 删除前取得服务退出码，防止启动崩溃却被报告为成功。
server_status="$(docker wait "$container_id")"
exit "$server_status"
