# HiCache 恢复实验

在 SGLang 上比较相同输入 P+S 的 Cold、GPU L1 命中和 CPU L2 恢复性能，校验缓存状态并保存测量证据。
P 是历史 Prefix，S 是新增 Suffix；`1K=1024 tokens`。

工程保留两个入口：

- `launch_server.sh`：创建容器、启动服务、保存启动配置和源码补丁。
- `run_experiment.py`：构造输入、准备缓存、发送请求、检查命中、统计和绘图。

[详细用法与指标](docs/usage.md) · [逐轮实验记录与代码版本](docs/experiments/README.md)

## 环境与快速开始

需要可用的 NVIDIA GPU、Docker、本机已有的镜像和模型；脚本不下载模型或镜像。
默认使用相邻 `../sglang` checkout、`/state/partition/hdwang/models/Qwen3-8B` 和本地 `lmsysorg/sglang:latest` 镜像。
启动时会将镜像标签固定为镜像 ID，源码与权重只读挂载。

先确认所选 GPU 空闲且可以正常运行 CUDA，然后在宿主机的终端 A 启动专用服务：

```bash
bash launch_server.sh --dry-run
GPU=3 RUN_ID=baseline-demo CONTAINER_NAME=hicache-demo bash launch_server.sh
```

脚本保持前台显示日志；按 Ctrl-C 停止本次容器。服务就绪后，在终端 B 做一次小规模流程验证：

```bash
docker exec --user "$(id -u):$(id -g)" hicache-demo \
  python /experiment/run_experiment.py run \
  --tokenizer /model --output-dir /results/baseline-demo-samples \
  --prefix-lengths 8k --suffix-lengths 128 --output-length 1 \
  --groups cold l1 --repeats 3
```

`run` 只连接已有服务。每次使用新的 `RUN_ID` 和测量目录，默认结果在宿主机 `results/`。
3 次用于检查流程；`--repeats` 指定每组需要收集的有效样本数，默认 20 次。
测量期间保持单请求、并发 1，服务只供本次实验使用。

## 常用参数

服务参数使用环境变量，额外参数直接追加给 SGLang：

```bash
GPU=3 MODEL_DIR=/path/to/model DTYPE=bfloat16 KV_DTYPE=bfloat16 \
  HICACHE=1 FORWARD_TIMING=1 RUN_ID=timing-demo \
  CONTAINER_NAME=hicache-timing bash launch_server.sh
```

| 参数 | 用途 |
|---|---|
| `GPU`、`MODEL_DIR`、`SGLANG_DIR`、`IMAGE` | 选择物理 GPU、模型、源码和本地镜像 |
| `DTYPE`、`KV_DTYPE` | 权重和 KV 精度，默认 BF16 |
| `CONTEXT_LENGTH`、`MAX_TOTAL_TOKENS`、`MEM_FRACTION` | 上下文、GPU KV 容量上限和静态显存预算 |
| `HICACHE=1`、`HICACHE_SIZE` | 开启 CPU L2 和本地缓存控制，CPU 池大小单位为十进制 GB |
| `FORWARD_TIMING=1` | 开启原生 CUDA Event forward 指标，需同时开启 HiCache |
| `PORT`、`RESULTS_DIR`、`RUN_ID`、`CONTAINER_NAME` | 服务端口和本次输出、容器标识 |

当前 Driver 只接受 TP=PP=DP=1、并发 1。默认 chunked prefill=4096、page size=1、Graph 关闭、overlap 开启。
L2 需要配套 SGLang 的本地实验补丁；[L2 操作与 Profiler 用法](docs/usage.md)说明了限制和验证步骤。

Driver 的主要参数：

```bash
python3 run_experiment.py run --help
# 在容器内执行 run 时，可组合指定：
# --prefix-lengths 8k 16k 32k --suffix-lengths 32 128 512 2048
# --groups cold l1 l2 --repeats 20 --gpu-timing
```

固定输入通过 P+S₀ 预热、P+S 测量，S₀ 与 S 从首个 token 分叉。每个候选样本重新准备缓存。
要求 Cold 各层命中为 0；L1 device=P、host=0；L2 device=0、host=P，且实际完成的恢复字节数正确。
混合命中、回撤、异常结束和输出不一致的尝试会保留并标为无效，不进入正式统计。

## 结果与重新分析

| 内容 | 保存位置 |
|---|---|
| 工程用法和指标定义 | 本文件、`docs/usage.md` |
| 各轮目的、配置、版本、结论与故障记录 | `docs/experiments/`，纳入 Git |
| 输入、CSV、图表、日志、源码快照和校验 | `results/<run-id>/`，不纳入 Git |

每轮结果包含 `config.json`、`inputs.json`、`responses.jsonl`、`samples.csv`、`summary.csv` 和 `source/`。
启动证据保存在 `launch.json`、`sglang.patch` 和 `server.log` 中。逐轮记录链接到对应的原始数据，不把历史结果合并为同一次运行。

```bash
# 不需要服务器或 GPU；绘图需要 matplotlib。
python3 run_experiment.py analyze results/baseline-demo-samples
# 只重新生成 CSV。
python3 run_experiment.py analyze results/baseline-demo-samples --no-plot
```

`latency.png` 使用对数纵轴；固定 S 扫描 P、固定 P 扫描 S 都受支持。
`timing.png` 对比 GPU 流区间；L1 同时扫描 P/S 时还生成 Median/P95 热力图。
启用环境监控时，`all_cache_valid_summary.csv` 另外保留所有缓存有效尝试的统计。

## 主要指标

延迟字段单位为 ms；完整字段、Profiler 区间和环境检查定义见 [详细用法](docs/usage.md#指标含义)。

| 字段 | 含义 |
|---|---|
| `client_latency_ms` | 非流式请求提交至响应读完，包含 HTTP、调度和推理 |
| `server_first_token_ms` | 服务端首 token 延迟，包含排队等开销 |
| `queue_ms` | 调度等待队列中的累计时间 |
| `kv_load_ms` | 请求期间已完成 H2D 操作的 GPU 流区间累计值 |
| `kv_load_tokens`、`kv_load_bytes` | 实际完成的 KV 恢复 token 数和 payload 字节数 |
| `prefill_forward_gpu_ms` | 原生 extend CUDA Event 区间；Cold 累加全部 chunk，L1/L2 计算新增 S |
| `cache_valid`、`environment_valid`、`valid` | 缓存正确性、可选环境检查、正式样本有效性 |

GPU 流区间可能包含等待和提交间隙；KV 恢复与 Prefill 可重叠，不能相加为请求延迟。
Median/P95 根据有效样本计算，P95 使用线性插值。缓存检查通过和监控覆盖都不能证明共享主机完全没有干扰。

## 本地检查

```bash
bash -n launch_server.sh
python3 run_experiment.py self-test
```

`self-test` 使用模拟 tokenizer、服务和监控，不需要 GPU，不能替代真机验证。
Prefix lookup 的独立 CPU 时间、逐层等待时长尚未单独导出，现有 CSV 不代表完整的服务端阶段分解。
