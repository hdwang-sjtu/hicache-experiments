#!/usr/bin/env bash
# 当前环境入口：直接启动 SGLang，或按命名配置启动；Docker 包装也调用这里。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
用法：bash serve.sh [配置名] [--dry-run] [SGLang 参数 ...]
  bash serve.sh --list-configs
  MODEL_ROOT=/models bash serve.sh qwen3-8b-4k
  MODEL_ROOT=/models bash serve.sh qwen3-8b-4k --mem-fraction-static 0.85
无配置名时兼容旧调用：MODEL_DIR=/path/to/model HICACHE=1 bash serve.sh。

命名配置位于 configs/serve/<名称>.yaml；CLI > 显式环境变量 > 文件。
  MODEL_ROOT / MODEL_DIR     模型根目录 / 完整模型目录覆盖；--model-path 优先级更高
  SGLANG_DIR                 源码目录；默认使用相邻 ../sglang（存在时）
                             显式 SGLANG_DIR='' 使用当前环境安装包
  PYTHON_BIN / GPU            默认 python3；GPU 可选，设置 CUDA_VISIBLE_DEVICES
  RESULTS_DIR / RUN_ID        启动记录根目录 / 单层目录名，默认项目 results / UTC 时间
服务环境变量仍支持 HOST、PORT、TP、DTYPE、KV_DTYPE、CONTEXT_LENGTH、MAX_TOTAL_TOKENS、
MEM_FRACTION、HICACHE、HICACHE_SIZE、FORWARD_TIMING。
旧模式默认上下文 40960、KV 池 36864、BF16、TP=1、显存比例 0.7、HiCache/计时关闭。
--dry-run 不访问模型或 GPU，不写结果；命名配置解析仅需要 PyYAML。
内部选项 --exec / --resolved-profile：由 Docker 包装保存记录并传入已解析配置。
HELP
    exit 0
fi
dry_run=false
record=true
profile=""
resolved_profile=""
# 启动器选项可放配置名前后；遇到首个 SGLang 参数后，剩余参数原样交给服务。
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --exec) record=false; shift ;;
        --resolved-profile) resolved_profile="${2:?缺少已解析配置路径}"; shift 2 ;;
        --list-configs) "$PYTHON_BIN" "$SCRIPT_DIR/serve_config.py" list; exit ;;
        --) shift; break ;;
        -*) break ;;
        *) [[ -z "$profile" ]] || { echo '只能选择一个配置名。' >&2; exit 2; }
           profile="$1"; shift ;;
    esac
done
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_.-]+$ && "$RUN_ID" != . && "$RUN_ID" != .. ]] || {
    echo 'RUN_ID 必须是单层目录名。' >&2; exit 2;
}
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
[[ "$RESULTS_DIR" == /* ]] || RESULTS_DIR="$PWD/$RESULTS_DIR"
run_dir="$RESULTS_DIR/$RUN_ID"
profile_json=""
if [[ -n "$resolved_profile" ]]; then
    if "$record" || [[ -n "$profile" || $# != 0 ]]; then
        echo '--resolved-profile 仅供 --exec 使用，不能再追加参数。' >&2; exit 2
    fi
    profile_json="$(cat "$resolved_profile")"
    profile_config_file="$(dirname "$resolved_profile")/server-config.yaml"
elif [[ -n "$profile" ]]; then
    "$record" || { echo '--exec 命名模式需要 Docker 提供已解析配置。' >&2; exit 2; }
    profile_json="$("$PYTHON_BIN" "$SCRIPT_DIR/serve_config.py" resolve --runtime native "$profile" -- "$@")"
    profile_config_file="$run_dir/server-config.yaml"
fi
if [[ -n "$profile_json" ]]; then
    # NUL 分隔保留空格、换行和 JSON；不 eval，也不 source YAML 内容。
    mapfile -d '' -t values < <("$PYTHON_BIN" "$SCRIPT_DIR/serve_config.py" unpack <<< "$profile_json")
    ((${#values[@]} >= 4)) || exit 2
    MODEL_DIR="${values[1]}"
    env_count="${values[3]}"
    runtime_env=("${values[@]:4:env_count}")
    profile_cli=("${values[@]:4+env_count}")
    server_args=("$PYTHON_BIN" -m sglang.launch_server --config "$profile_config_file" "${profile_cli[@]}")
else
    # 无配置名保留旧默认值；这些值绝不追加到命名配置上。
    MODEL_DIR="${MODEL_DIR:?请设置 MODEL_DIR 为模型目录，或选择命名配置并设置 MODEL_ROOT}"
    HICACHE="${HICACHE:-0}"
    FORWARD_TIMING="${FORWARD_TIMING:-0}"
    [[ "$HICACHE" =~ ^[01]$ && "$FORWARD_TIMING" =~ ^[01]$ ]] || {
        echo 'HICACHE 和 FORWARD_TIMING 必须为 0 或 1。' >&2; exit 2;
    }
    if [[ "$FORWARD_TIMING" == 1 && "$HICACHE" != 1 ]]; then
        echo 'FORWARD_TIMING=1 需要 HICACHE=1，以便测量前后确认服务空闲。' >&2; exit 2
    fi
    runtime_env=("HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}"
        "SGLANG_ENABLE_METRICS_DEVICE_TIMER=$FORWARD_TIMING")
    [[ "$HICACHE" == 0 ]] || runtime_env+=("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python")
    server_args=("$PYTHON_BIN" -m sglang.launch_server
        --model-path "$MODEL_DIR" --host "${HOST:-127.0.0.1}" --port "${PORT:-39081}"
        --tp-size "${TP:-1}" --dtype "${DTYPE:-bfloat16}" --kv-cache-dtype "${KV_DTYPE:-bfloat16}"
        --context-length "${CONTEXT_LENGTH:-40960}" --max-total-tokens "${MAX_TOTAL_TOKENS:-36864}"
        --mem-fraction-static "${MEM_FRACTION:-0.7}"
        --max-running-requests 1 --chunked-prefill-size 4096 --page-size 1
        --enable-metrics --return-output-ids
        --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled)
    if [[ "$HICACHE" == 1 ]]; then
        server_args+=(--enable-hierarchical-cache --enable-hicache-experiment
            --hicache-host-memory-mode cache --hicache-write-policy write_through
            --hicache-size "${HICACHE_SIZE:-8}" --hicache-io-backend kernel --hicache-mem-layout page_first)
    fi
    server_args+=("$@")
fi
# 显式空字符串使用安装包；仅未设置时自动寻找相邻源码。
if [[ ! -v SGLANG_DIR ]]; then
    SGLANG_DIR=""
    [[ ! -d "$SCRIPT_DIR/../sglang/python/sglang" ]] || SGLANG_DIR="$SCRIPT_DIR/../sglang"
fi
if [[ -n "$SGLANG_DIR" ]]; then
    if ! "$dry_run"; then
        [[ -d "$SGLANG_DIR/python/sglang" ]] || { echo 'SGLANG_DIR 中没有 python/sglang。' >&2; exit 2; }
        SGLANG_DIR="$(cd "$SGLANG_DIR" && pwd)"
    fi
    runtime_env+=("PYTHONPATH=$SGLANG_DIR/python${PYTHONPATH:+:$PYTHONPATH}")
fi
[[ -z ${GPU:-} ]] || runtime_env+=("CUDA_VISIBLE_DEVICES=$GPU")
if "$dry_run"; then
    [[ -z "$profile_json" ]] || "$PYTHON_BIN" "$SCRIPT_DIR/serve_config.py" preview <<< "$profile_json"
    printf '%q ' env "${runtime_env[@]}" "${server_args[@]}"
    printf '\n'
    exit 0
fi
[[ -f "$MODEL_DIR/config.json" ]] || { echo "模型目录中没有 config.json：$MODEL_DIR" >&2; exit 2; }
export "${runtime_env[@]}"
if "$record"; then
    mkdir -p "$RESULTS_DIR"
    mkdir "$run_dir" # 不覆盖已有启动记录。
    metadata_args=()
    [[ -z "$profile_json" ]] || metadata_args+=(--profile-json "$profile_json")
    "$PYTHON_BIN" "$SCRIPT_DIR/launch_metadata.py" \
        --directory "$run_dir" --runtime native --model-dir "$MODEL_DIR" \
        --source-dir "$SGLANG_DIR" --gpu "${CUDA_VISIBLE_DEVICES:-inherited}" \
        "${metadata_args[@]}" -- "${server_args[@]}"
    echo "启动记录：$run_dir；客户端可用 --server-artifacts 指向该目录或其本地副本。"
    # exec 保留服务退出码和信号语义；长期任务请在 tmux 等会话中启动。
    exec "${server_args[@]}" > >(tee "$run_dir/server.log") 2>&1
fi
exec "${server_args[@]}"
