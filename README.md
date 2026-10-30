# HiCache 恢复实验

两个脚本：`launch_server.sh` 启动服务；`run_experiment.py` 构造输入、运行请求、检查命中、保存结果和绘图。

**当前实现阶段 1：Cold / GPU L1。尚未实现 CPU L2 定向 Demote、逐请求 H2D 统计和 GPU Prefill 埋点。**
本阶段关闭 HiCache；后续正式三组比较需要在相同 HiCache 设置下重新测量。

## 怎么读这两个脚本

`launch_server.sh` 从上到下分为四步，代码中已用中文标出：读取参数 → 组装服务与 Docker 命令 → 保存复现信息 → 启动容器并跟踪日志。
配置模型、GPU、精度或显存预算时，先看开头的环境变量和 `server_args`；修改挂载时看 `docker_args`。

`run_experiment.py` 可以按下面的调用顺序阅读：

```text
main()                       解析 run / analyze / self-test 子命令
└── run()                    准备目录和输入，等待并检查服务，保存最终状态
    ├── make_cases()         构造精确的 P、S、S₀，保证只共享 P 个 token
    ├── check_server()       检查单卡、单请求、上下文长度和 KV 容量
    ├── run_samples()        执行运行预热，收集每组所需的有效测量
    │   └── execute_sample() 清空缓存 → 按组准备 → 测量 → 校验
    │       ├── Client      发送 HTTP 请求并保存原始响应
    │       └── validate()  检查长度、各层命中、回撤和时间字段
    └── analyze()            过滤无效样本，输出 Median、P95 和图表
```

有两种容易混淆的“预热”：`warmup` 是每个长度组合开始时的运行预热，不写入测量 CSV；`prepare` 是每一次 L1 样本都必须重新做的 P+S₀ 请求，用来构造缓存状态。
`measure` 才是计入 CSV 的 P+S 请求，且其清空和准备耗时不计入客户端测量延迟。

函数说明、关键流程和命令行帮助使用中文。变量、CSV 字段和图表标签保留英文；内置测试语料也保持原文，以免修改注释时改变输入。
可用 `bash launch_server.sh --help` 和 `python3 run_experiment.py run --help` 查看中文参数说明。

## 运行

在宿主机的普通终端运行，需要可以访问 Docker 和 NVIDIA GPU。脚本不会下载镜像或模型。
默认使用当前机器的 Qwen3-8B 与相邻 SGLang checkout：

```bash
cd /home/hdwang/Workspace/sglang_prj/hicache-experiments
bash launch_server.sh --dry-run

# 先用 nvidia-smi 确认 GPU 1 空闲。RUN_ID 每次启动需不同。
GPU=1 RUN_ID=baseline-8k bash launch_server.sh
```

启动脚本保持前台显示并保存服务日志，打印第二个终端应执行的完整命令。按 Ctrl-C 停止本次创建的容器。
它将本地镜像标签解析为 immutable image ID 后启动，避免测试期间标签变化。

第二个终端执行脚本打印的命令，例如：

```bash
docker exec --user "$(id -u):$(id -g)" hicache-exp-$(id -u)-baseline-8k \
  python /experiment/run_experiment.py run \
  --base-url http://127.0.0.1:39081 --tokenizer /model \
  --output-dir /results/baseline-8k \
  --prefix-lengths 8k --suffix-lengths 128 --repeats 3
```

Driver 用宿主机 UID 写结果，适用于本机 NFS home。权重和源码只读挂载。
Driver 会等待 `/health` 就绪；不要同时向这个服务发送其他请求。每个样本会调用 `/flush_cache`，只应使用专用实验服务。

8K 的命中验证通过后，在同一服务上换一个结果目录扫描：

```bash
docker exec --user "$(id -u):$(id -g)" hicache-exp-$(id -u)-baseline-8k \
  python /experiment/run_experiment.py run \
  --tokenizer /model --output-dir /results/baseline-scan \
  --prefix-lengths 8k 16k 32k --suffix-lengths 128 --repeats 20
```

也可以设置 `--suffix-lengths 32 128 512 2048`、`--output-length 1`、`--groups cold l1`、`--seed 0`。
`8k` 表示 8192 tokens。Qwen3-8B 的配置上限是 40960，P + S + 输出必须在上限内；本阶段不做 64K 或 RoPE 扩展。

## 参数与结果

启动参数通过环境变量修改，额外命令行参数直接追加给 SGLang：

```bash
GPU=1 MODEL_DIR=/path/to/local/model DTYPE=bfloat16 KV_DTYPE=bfloat16 \
  RUN_ID=another-model bash launch_server.sh
```

可覆盖 `SGLANG_DIR`、`IMAGE`、`PORT`、`CONTEXT_LENGTH`、`MAX_TOTAL_TOKENS`、`MEM_FRACTION`、`RESULTS_DIR`。
`TP` 是启动参数；目前 Driver 为了保证实验归属，仅接受 TP=PP=DP=1。
默认上下文 40960、KV 容量上限 36864 tokens、chunk size 4096、并发 1、Graph 关闭、overlap 保留。
其他模型需自行设置正确上下文长度，dtype 必须由模型及所用后端支持。

每次运行保存到 `results/<RUN_ID>/`：

- `launch.json`、`sglang.patch`、`gpu.txt`、`server.log`：启动参数、镜像 ID、源码版本和服务证据。
- `config.json`：Driver 参数、状态、实际 `/server_info`、模型信息、依赖版本、tokenizer 哈希。
- `inputs.json`：固定文本、精确 prefix / suffix / 预热 suffix IDs 和测量输入哈希。
- `responses.jsonl`：控制操作、预热和测量的原始响应，阶段分别标记。
- `samples.csv`：每个测量样本及有效性；`summary.csv`：有效样本 Median / P95 和无效计数。
- `latency.png`：服务端首 token 延迟图（需要 matplotlib）；`source/` 保存两个脚本的运行版本。

目录不得复用。若 Driver 中断或出错，保留已有数据并标为 `failed` / `interrupted`；不会把不完整的组伪装成完成。
一个服务可对应多个结果目录，后续目录会关联原启动信息并复制当时的日志快照。
直接对外部服务运行也可以，但没有启动记录时无法证明镜像与源码版本，会明确提示。

运行和重新分析使用相同 Python 脚本：

```bash
# 在装有 matplotlib 的 Python 环境中；不需要 GPU、tokenizer 或服务器。
python3 run_experiment.py analyze results/baseline-8k
# 只重新生成 CSV，无第三方依赖。
python3 run_experiment.py analyze results/baseline-8k --no-plot
```

请求执行需要 `transformers` 的本地 tokenizer；建议使用服务容器内已有环境。
绘图缺少 matplotlib 时 CSV 仍然生成，会提示补装后重新分析；也可显式传 `--no-plot`。

## 样本如何构造和判定

默认把脚本内固定的英文实验笔记分词后重复、截取到精确长度。这是合成性能输入，不用于评价模型回答质量。
可以用 `--text-file /path/to/text.txt` 指定固定 UTF-8 文本；特殊 token 不允许进入测试文本。

每个 P/S 组合先执行一轮两组运行预热，不计入 CSV。每次测量分别执行：

1. Cold：确认缓存清空成功 → 发送 P+S。
2. L1：确认缓存清空成功 → 发送 P+S₀ 生成 1 token → 发送相同测量输入 P+S。

S₀ 和 S 的第一个 token 保证不同。每次重复都重新准备，避免上次的 S 也被缓存。
每轮随机化组顺序（种子可复现），单请求串行执行。

有效样本必须满足：

- 输入 tokens = P+S，输出 tokens = 指定值，正常到达长度限制；temperature=0、ignore_eos=true。
- Cold 总命中 / device / host / storage 均为 0。
- L1 总命中和 device 恰好为 P，host / storage 为 0。
- 回撤次数为 0；服务端必须返回完整、有效的时间统计。

服务端在零命中时可能返回 `cached_tokens_details=null`，只有同时 `cached_tokens=0` 才解释为各层零命中。
其他缺失字段不默认为零。部分、混合和超出 P 的命中会保留为无效样本，不进入分位数。
每组收集指定有效次数，最多尝试 `3 * repeats` 次，可用 `--max-attempts` 调整。预热、HTTP 或清空失败会终止，避免在未知状态下继续测量。
脚本校验容量、页对齐及服务设置；它不能阻止其他客户端插入请求，需保持实验服务独占。

## 指标含义

所有 CSV 延迟字段单位为 ms：

| 字段 | 含义 |
|---|---|
| `client_latency_ms` | 客户端提交非流式请求至读完响应，包含 HTTP、排队和推理，不是 TTFT |
| `server_first_token_ms` | 服务端已有的首 token 延迟，包含服务端处理和排队，不是纯 GPU Prefill |
| `server_e2e_ms` | 服务端已有的完整请求延迟，输出多个 tokens 时包含 decode |
| `queue_ms` | 请求在调度等待队列中的累计时间 |
| `forward_entry_time` / `prefill_finished_time` | 原样保存的服务端时间戳（秒）；不当作 CUDA Event 时间 |

Median 和 P95 仅对有效样本计算；P95 使用排序后线性插值。3 次用于流程验证，20 次是首轮最低样本数，不能据此推断稳定的尾延迟。
图只绘制已达到目标有效次数的组，同时标记整个运行状态。

`/flush_cache` 在当前 SGLang 中也会调用 device allocator 的 `empty_cache()`。
因此本阶段 Cold/L1 差异可能包含瞬态分配状态差异，是缓存基线检查；不能把差值当作纯计算量，也不能外推为 KV 加载时间。

## 本地检查和后续阶段

```bash
bash -n launch_server.sh
python3 run_experiment.py self-test
```

`self-test` 不需要第三方包或 GPU，检查精确分叉、每次重置、预热隔离、异常命中、配置限制和无效样本过滤。它不代表真实 GPU 测试。

阶段 1 真机通过后，再增加 L2 控制：等待备份完成、Scheduler 安全点定向 Demote、保留 host 副本及索引。
随后复用 SGLang 已有的加载字节数和 CUDA Event 指标，补齐请求关联及必要计时，最后通过 Profiler 验证加载和计算重叠。
