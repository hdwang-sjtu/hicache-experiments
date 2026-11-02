#!/usr/bin/env bash
# 当前环境入口：统一 SGLang 参数。Docker 包装脚本也调用本文件。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
用法：MODEL_DIR=/path/to/model [环境变量=值 ...] bash serve.sh [--dry-run] [SGLang 参数 ...]
直接使用已有 Python/SGLang 环境，不调用 Docker，不下载模型。
  MODEL_DIR                  必填：当前环境的完整模型目录
  SGLANG_DIR                 源码目录；默认使用相邻 ../sglang（存在时）
                             显式 SGLANG_DIR='' 使用已安装的包
  PYTHON_BIN / GPU            默认 python3；GPU 可选，设置 CUDA_VISIBLE_DEVICES
  HOST / PORT                默认 127.0.0.1 / 39081
  RESULTS_DIR / RUN_ID        启动记录根目录 / 单层目录名，默认项目 results / UTC 时间
  TP / DTYPE / KV_DTYPE       默认 1 / bfloat16 / bfloat16
  CONTEXT_LENGTH             默认 40960
  MAX_TOTAL_TOKENS            默认 36864
  MEM_FRACTION               默认 0.7
  HICACHE / HICACHE_SIZE      默认 0 / 8（十进制 GB）；开启后需配套实验补丁
  FORWARD_TIMING              默认 0；设为 1 开启原生 GPU 计时，要求 HICACHE=1
额外参数直接追加给 SGLang。--dry-run 不访问模型、GPU 或 Python 依赖。
内部选项 --exec：由 Docker 包装脚本保存记录，本脚本只 exec 服务。
HELP
    exit 0
fi
dry_run=false
record=true
case ${1:-} in
    --dry-run) dry_run=true; shift ;;
    --exec) record=false; shift ;;
esac
[[ ${1:-} != -- ]] || shift
MODEL_DIR="${MODEL_DIR:?请设置 MODEL_DIR 为模型目录}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HICACHE="${HICACHE:-0}"
FORWARD_TIMING="${FORWARD_TIMING:-0}"
[[ "$HICACHE" =~ ^[01]$ && "$FORWARD_TIMING" =~ ^[01]$ ]] || {
    echo 'HICACHE 和 FORWARD_TIMING 必须为 0 或 1。' >&2; exit 2;
}
if [[ "$FORWARD_TIMING" == 1 && "$HICACHE" != 1 ]]; then
    echo 'FORWARD_TIMING=1 需要 HICACHE=1，以便测量前后确认服务空闲。' >&2
    exit 2
fi
# 显式空字符串使用安装包；仅未设置时自动寻找相邻源码。
if [[ ! -v SGLANG_DIR ]]; then
    SGLANG_DIR=""
    [[ ! -d "$SCRIPT_DIR/../sglang/python/sglang" ]] || SGLANG_DIR="$SCRIPT_DIR/../sglang"
fi
runtime_env=(
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}"
    "SGLANG_ENABLE_METRICS_DEVICE_TIMER=$FORWARD_TIMING"
)
if [[ -n "$SGLANG_DIR" ]]; then
    if ! "$dry_run"; then
        [[ -d "$SGLANG_DIR/python/sglang" ]] || { echo 'SGLANG_DIR 中没有 python/sglang。' >&2; exit 2; }
        SGLANG_DIR="$(cd "$SGLANG_DIR" && pwd)"
    fi
    runtime_env+=("PYTHONPATH=$SGLANG_DIR/python${PYTHONPATH:+:$PYTHONPATH}")
fi
[[ -z ${GPU:-} ]] || runtime_env+=("CUDA_VISIBLE_DEVICES=$GPU")
[[ "$HICACHE" == 0 ]] || runtime_env+=("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python")
server_args=("$PYTHON_BIN" -m sglang.launch_server
    --model-path "$MODEL_DIR" --host "${HOST:-127.0.0.1}" --port "${PORT:-39081}"
    --tp-size "${TP:-1}" --dtype "${DTYPE:-bfloat16}" --kv-cache-dtype "${KV_DTYPE:-bfloat16}"
    --context-length "${CONTEXT_LENGTH:-40960}" --max-total-tokens "${MAX_TOTAL_TOKENS:-36864}"
    --mem-fraction-static "${MEM_FRACTION:-0.7}"
    --max-running-requests 1 --chunked-prefill-size 4096 --page-size 1
    --enable-metrics --return-output-ids
    --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
)
if [[ "$HICACHE" == 1 ]]; then
    server_args+=(--enable-hierarchical-cache --enable-hicache-experiment
        --hicache-host-memory-mode cache --hicache-write-policy write_through
        --hicache-size "${HICACHE_SIZE:-8}" --hicache-io-backend kernel --hicache-mem-layout page_first)
fi
server_args+=("$@")
if "$dry_run"; then
    printf '%q ' env "${runtime_env[@]}" "${server_args[@]}"
    printf '\n'
    exit 0
fi
[[ -f "$MODEL_DIR/config.json" ]] || { echo 'MODEL_DIR 中没有 config.json。' >&2; exit 2; }
export "${runtime_env[@]}"
if "$record"; then
    RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
    [[ "$RUN_ID" =~ ^[a-zA-Z0-9_.-]+$ && "$RUN_ID" != . && "$RUN_ID" != .. ]] || {
        echo 'RUN_ID 必须是单层目录名。' >&2; exit 2;
    }
    RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
    mkdir -p "$RESULTS_DIR"
    run_dir="$RESULTS_DIR/$RUN_ID"
    mkdir "$run_dir" # 不覆盖已有启动记录。
    "$PYTHON_BIN" "$SCRIPT_DIR/launch_metadata.py" \
        --directory "$run_dir" --runtime native --model-dir "$MODEL_DIR" \
        --source-dir "$SGLANG_DIR" --gpu "${CUDA_VISIBLE_DEVICES:-inherited}" \
        -- "${server_args[@]}"
    echo "启动记录：$run_dir；客户端可用 --server-artifacts 指向该目录或其本地副本。"
    # exec 保留服务退出码和信号语义；长期任务请在 tmux 等会话中启动。
    exec "${server_args[@]}" > >(tee "$run_dir/server.log") 2>&1
fi
exec "${server_args[@]}"
