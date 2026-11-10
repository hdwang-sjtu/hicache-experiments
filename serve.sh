#!/usr/bin/env bash
# 当前环境入口：直接启动 SGLang，或按命名配置启动；Docker 包装也调用这里。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
用法：bash serve.sh <配置名> [--dry-run] [SGLang 参数 ...]
  bash serve.sh --list-configs
  MODEL_DIR=/models/Qwen3-8B bash serve.sh qwen3-8b-smoke
  MODEL_DIR=/models/Qwen3-8B bash serve.sh qwen3-8b-smoke --mem-fraction-static 0.85
必须选择配置名；服务默认值统一保存在 configs/experiments/。

命名配置位于 configs/experiments/<名称>.yaml；CLI > 显式环境变量 > 文件。
  MODEL_DIR     完整模型/配置目录；--model-path 优先级更高
  SGLANG_DIR                 源码目录；默认使用相邻 ../sglang（存在时）
                             显式 SGLANG_DIR='' 使用当前环境安装包
  PYTHON_BIN / GPU            默认 python3；GPU 可选，设置 CUDA_VISIBLE_DEVICES
  RESULTS_DIR / RUN_ID        启动记录根目录 / 单层目录名，默认项目 results / UTC 时间
服务环境变量仍支持 HOST、PORT、TP、EP、CHUNKED_PREFILL_SIZE、DTYPE、KV_DTYPE、CONTEXT_LENGTH、MAX_TOTAL_TOKENS、
MEM_FRACTION、HICACHE、HICACHE_SIZE、FORWARD_TIMING。
--dry-run 不访问模型或 GPU，不写结果；命名配置解析仅需要 PyYAML。
内部选项 --exec / --resolved-profile：总控或 Docker 传入已解析配置，避免重复覆盖。
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
        --list-configs) "$PYTHON_BIN" "$SCRIPT_DIR/hicache/config.py" list; exit ;;
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
    if [[ -n "$profile" || $# != 0 ]]; then
        echo '已解析配置不能再追加配置名或参数。' >&2; exit 2
    fi
    profile_json="$(cat "$resolved_profile")"
    if "$record"; then
        profile_config_file="$run_dir/server-config.yaml"
    else
        profile_config_file="$(dirname "$resolved_profile")/server-config.yaml"
    fi
elif [[ -n "$profile" ]]; then
    "$record" || { echo '--exec 命名模式需要 Docker 提供已解析配置。' >&2; exit 2; }
    profile_json="$("$PYTHON_BIN" "$SCRIPT_DIR/hicache/config.py" resolve --runtime native "$profile" -- "$@")"
    profile_config_file="$run_dir/server-config.yaml"
else
    echo '请指定 configs/experiments 下的配置名；用 --list-configs 查看。' >&2; exit 2
fi
# 配置已完整解析；这里只把数据转成参数数组，不维护另一套服务默认值。
# NUL 分隔保留空格、换行和 JSON；不 eval，也不 source YAML 内容。
mapfile -d '' -t values < <("$PYTHON_BIN" "$SCRIPT_DIR/hicache/config.py" unpack <<< "$profile_json")
((${#values[@]} >= 4)) || exit 2
MODEL_DIR="${values[1]}"
env_count="${values[3]}"
runtime_env=("${values[@]:4:env_count}")
profile_cli=("${values[@]:4+env_count}")
server_args=("$PYTHON_BIN" -m sglang.launch_server --config "$profile_config_file" "${profile_cli[@]}")
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
    "$PYTHON_BIN" "$SCRIPT_DIR/hicache/config.py" preview <<< "$profile_json"
    printf '%q ' env "${runtime_env[@]}" "${server_args[@]}"
    printf '\n'
    exit 0
fi
[[ -f "$MODEL_DIR/config.json" ]] || { echo "模型目录中没有 config.json：$MODEL_DIR" >&2; exit 2; }
export "${runtime_env[@]}"
if "$record"; then
    mkdir -p "$RESULTS_DIR"
    mkdir "$run_dir" # 不覆盖已有启动记录。
    "$PYTHON_BIN" "$SCRIPT_DIR/hicache/metadata.py" \
        --directory "$run_dir" --runtime native --model-dir "$MODEL_DIR" \
        --source-dir "$SGLANG_DIR" --gpu "${CUDA_VISIBLE_DEVICES:-inherited}" \
        --profile-json "$profile_json" -- "${server_args[@]}"
    echo "启动记录：$run_dir；客户端可用 --server-artifacts 指向该目录或其本地副本。"
    # exec 保留服务退出码和信号语义；长期任务请在 tmux 等会话中启动。
    exec "${server_args[@]}" > >(tee "$run_dir/server.log") 2>&1
fi
exec "${server_args[@]}"
