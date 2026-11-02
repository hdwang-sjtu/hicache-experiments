# HiCache 恢复实验

比较相同输入 P+S 的 Cold、GPU L1 命中和 CPU L2 恢复性能。P 是历史 Prefix，S 是新增 Suffix；`1K=1024 tokens`。

部署与实验分开，三个入口各自负责一件事：

| 入口 | 职责 | 在哪里运行 |
|---|---|---|
| `launch_server.sh` | 创建 Docker 容器，挂载模型/源码，调用 `serve.sh`，管理本次容器 | 有 Docker 和 NVIDIA runtime 的宿主机 |
| `serve.sh` | 统一 SGLang 参数，在当前 Python 环境启动服务 | 已有环境的服务器或云容器内部 |
| `run_experiment.py` | 通过 HTTP 构造缓存状态、测量、校验、统计和绘图 | 任意能连接服务的客户端 |

服务参数可保存为 `configs/serve/<名称>.yaml`；`serve_config.py` 和 `launch_metadata.py` 是共用的内部解析/记录工具。
[部署与 Vast.ai 迁移](docs/deployment.md) · [实验用法与指标](docs/usage.md) · [逐轮实验记录](docs/experiments/README.md)

## 1. 启动服务：二选一

先准备完整模型目录。L2 和 GPU 计时需要带本地实验补丁的 SGLang；官方镜像本身不包含该补丁。
使用当前仓库的相邻 `../sglang` 时会自动加载该源码，也可显式设置 `SGLANG_DIR`。

每台机器先设置模型根目录，命名配置只需要轻量 YAML 依赖：

```bash
python3 -m pip install -r requirements-launch.txt
export MODEL_ROOT=/workspace/models
bash serve.sh --list-configs
```

已有环境（包括 Vast 普通容器）直接启动；需要 Docker 时使用另一个入口，二选一：

```bash
# 当前环境，模型目录自动解析为 $MODEL_ROOT/Qwen3-8B。
RUN_ID=server-demo bash serve.sh qwen3-8b-4k

# Docker：镜像须提前下载，宿主机需有 NVIDIA runtime。
GPU=0 RUN_ID=server-demo CONTAINER_NAME=hicache-demo \
  bash launch_server.sh qwen3-8b-4k
```

该配置开启 HiCache 和 GPU 计时，使用 BF16、TP=1、总上下文 4096、KV 池 8192、CPU 池 8 GB。
两种方式均监听 `127.0.0.1:39081`，chunk=4096、page size=1、Graph 关闭、overlap 开启。
启动记录保存在 `results/server-demo/`。长期运行可放在 `tmux` 中；Docker 入口退出时清理其创建的容器。

```bash
# 临时覆盖：命令行 > 显式环境变量 > 配置文件。
bash serve.sh qwen3-8b-4k --mem-fraction-static 0.85
bash serve.sh qwen3-8b-4k --dry-run
```

完整规则与新增配置示例见 [命名服务配置](docs/deployment.md#命名服务配置)。不带配置名的旧环境变量调用继续支持。

## 2. 客户端执行实验

客户端不需要模型权重、SGLang、Docker 或 GPU；需要与服务端匹配的 tokenizer 文件。
服务和客户端在同一环境时可直接把完整模型目录作为 tokenizer 目录。

```bash
python3 -m pip install -r requirements-client.txt
python3 run_experiment.py run \
  --base-url http://127.0.0.1:39081 \
  --tokenizer /path/to/Qwen3-8B-tokenizer \
  --output-dir results/samples-demo \
  --prefix-lengths 2k --suffix-lengths 128 \
  --groups cold l1 l2 --output-length 1 --repeats 3 --gpu-timing
```

也可在上述 Docker 容器内运行，省去客户端环境安装：

```bash
docker exec --user "$(id -u):$(id -g)" hicache-demo \
  python /experiment/run_experiment.py run \
  --base-url http://127.0.0.1:39081 --tokenizer /model \
  --output-dir /results/samples-demo \
  --prefix-lengths 2k --suffix-lengths 128 \
  --groups cold l1 l2 --repeats 3 --gpu-timing
```

这里用 P=2K 验收：配置名中的 4K 是总上下文，必须容纳 P、S 和输出。
多项 P/S 自动组成矩阵；每个组合先预热，再收集指定数量的有效样本，每次重新准备缓存。
仅测 L1 时设置 `--groups l1`。服务应专供本轮实验，单请求、并发 1。
远程机器通过 [SSH 隧道](docs/deployment.md#远程客户端) 使用相同 Driver；当前缓存控制接口只接受服务端 loopback 连接。

## 常用参数

| 配置 | 位置 | 含义 |
|---|---|---|
| `MODEL_ROOT`、`MODEL_DIR`、`SGLANG_DIR` | 两个启动入口 | 模型根目录、完整模型目录覆盖、源码 checkout；`SGLANG_DIR=''` 使用安装包 |
| `GPU` | 启动入口 | Docker 选择物理卡；直接启动设置可见卡，省略则继承环境 |
| `DTYPE`、`KV_DTYPE`、`TP` | `serve.sh` | 权重/KV 精度与 TP；当前 Driver 仅支持 TP=PP=DP=1 |
| `CONTEXT_LENGTH`、`MAX_TOTAL_TOKENS`、`MEM_FRACTION` | 启动环境，可覆盖配置 | 无配置名时默认 40960、36864、0.7 |
| `HICACHE`、`HICACHE_SIZE`、`FORWARD_TIMING` | 启动环境，可覆盖配置 | 无配置名时默认 0、8 GB（十进制）、0；GPU 计时需开启 HiCache |
| `--base-url`、`--tokenizer`、`--output-dir` | Driver | 服务地址、客户端 tokenizer、客户端结果目录 |
| `--server-artifacts` | Driver，可选 | 客户端可读的启动记录目录或其副本，不要求共享磁盘 |
| `--profile --server-profile-dir /path` | Driver，诊断用 | 在服务端指定路径写 trace；正式扫描关闭 Profiler |

额外 SGLang 参数直接追加给任一启动入口。`--help` 查看用法，`--dry-run` 检查最终命令而不启动服务。

## 输出与当前边界

`config.json`、`inputs.json`、`responses.jsonl`、`samples.csv`、`summary.csv`、图表和 Driver 快照保存在测量目录。
可选启动记录包括源码版本、补丁、镜像 ID、启动命令和日志。实验记录写在 `docs/experiments/`；原始数据放 `results/`，不纳入 Git。

```bash
python3 run_experiment.py analyze results/samples-demo
# 无绘图库也可重新汇总 CSV：
python3 run_experiment.py analyze results/samples-demo --no-plot
```

命中、长度、回撤、输出和实际 H2D 传输量均需满足检查；异常尝试保留但不计入统计。
计时单位为 ms。`prefill_forward_gpu_ms` / `kv_load_ms` 是 GPU 流区间，可包含等待并互相重叠；客户端延迟包含网络。
Median/P95 基于有效样本，P95 使用线性插值；[完整定义](docs/usage.md#指标含义)。

当前支持自动矩阵采样、校验和绘图；服务启动、可选监控、断点续跑和云实例计费回收没有合成一键编排。
Prefix lookup 独立时间尚未导出。真实 GPU 和新云平台需各自验收，CPU 自测不能替代。

```bash
bash -n launch_server.sh serve.sh
python3 run_experiment.py self-test
python3 -m unittest discover -s tests -v
```
