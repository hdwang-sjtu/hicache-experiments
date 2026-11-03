# 部署与迁移

[工程入口](../README.md) · [项目架构](architecture.md) · [一键云任务](cloud-task.md) · [实验用法与指标](usage.md)

## 职责与目录

文件职责、调用者及目录用途统一见 [项目架构](architecture.md)；本页保留部署步骤、配置覆盖规则和操作命令。

SGLang 和模型放在工程外。默认自动使用相邻 `../sglang`；也可设置 `SGLANG_DIR=/path/to/sglang`。
显式 `SGLANG_DIR=''` 表示使用 Python 环境/镜像已经安装的包。安装包仍需与实验接口、指标和启动参数兼容。
CPU L2 定向逐出补丁在 SGLang 提交 `02a56e2b24` 中，官方上游镜像不会自动包含它。
客户端只需 `run_experiment.py` 和 tokenizer 文件；`--tokenizer` 永远是客户端路径，`--output-dir` 永远写在客户端。

## 命名服务配置

两种入口共用 `configs/serve/<配置名>.yaml`。解析配置只依赖 PyYAML，不需要在 Docker 宿主机安装 SGLang/Torch：

```bash
python3 -m pip install -r requirements-launch.txt
export MODEL_ROOT=/workspace/models
bash serve.sh --list-configs
bash serve.sh qwen3-8b-4k --dry-run
bash launch_server.sh qwen3-8b-4k --dry-run
```

预览不会读取模型、启动容器、访问 GPU 或创建结果目录；列举配置不依赖 PyYAML。
`--dry-run` 可放配置名前后，应放在追加的 SGLang 参数之前。实际启动时去掉它。

配置包含三个部分，完整示例见 [qwen3-8b-4k.yaml](../configs/serve/qwen3-8b-4k.yaml)：

```yaml
model: Qwen3-8B
env:
  SGLANG_ENABLE_METRICS_DEVICE_TIMER: "1"
  SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND: python
sglang:
  context-length: 4096
  max-total-tokens: 8192
  dtype: bfloat16
  kv-cache-dtype: bfloat16
  enable-hierarchical-cache: true
  hicache-size: 8
```

上面只展示结构，正式文件还包括实验接口、write-through、指标和 Graph 等设置。
`sglang` 使用原生连字符参数名，布尔值写 `true/false`，JSON 参数可写成 YAML 映射；由 SGLang 原生 `--config` 解析。
`model` 是相对 `MODEL_ROOT` 的目录名；配置不包含 GPU、镜像或本机绝对模型路径。`env` 接受 `SGLANG_*` 和两个离线开关。
每个文件是一份完整配置，不提供继承/组合。新增场景时复制示例、修改内容，并使用新的文件名（调用时省略 `.yaml`）。

覆盖顺序为 **命令行 > 显式环境变量 > 文件**，未设置的环境变量不会用旧脚本默认值覆盖配置。
原有 `HOST/PORT/TP/DTYPE/KV_DTYPE/CONTEXT_LENGTH/MAX_TOTAL_TOKENS/MEM_FRACTION/HICACHE_SIZE` 映射到对应 SGLang 参数；
`HICACHE` 同时控制 hierarchical-cache 和实验接口，`FORWARD_TIMING` 覆盖 GPU 计时开关。
关闭 HiCache 时也要关闭依赖它的计时，例如 `HICACHE=0 FORWARD_TIMING=0 bash serve.sh qwen3-8b-4k`。
SGLang 的启用标志不能写成 `--enable-hierarchical-cache false`；临时关闭使用上述环境变量，或修改 YAML 中的布尔值。

```bash
MEM_FRACTION=0.75 bash serve.sh qwen3-8b-4k --mem-fraction-static 0.85
# 最终显存比例为 0.85；环境覆盖和 CLI 都保留在启动记录中。

MODEL_DIR=/another/path/Qwen3-8B bash serve.sh qwen3-8b-4k
GPU=0 bash launch_server.sh qwen3-8b-4k --model-path /another/path/Qwen3-8B
```

模型路径顺序为 `--model-path`（含原生别名 `--model`）> `MODEL_DIR` > `MODEL_ROOT/model`。
CLI/`MODEL_DIR` 的相对路径相对于启动工作目录；Docker 在宿主机解析后挂载到 `/model`，不会向容器透传宿主机路径。
命名模式不能再叠加另一个 `--config`。GPU、镜像、源码与结果目录仍用原有部署变量。

每次启动保存 `profile.yaml` 原文、`server-config.yaml`（环境覆盖后的实际下发 YAML）、`profile-resolution.json`（路径、运行时环境和 CLI）及 `launch.json` 中的名称/哈希。
CLI 仍可覆盖下发 YAML，最终命令与覆盖参数一同记录；实际服务配置由 Driver 保存的 `/server_info` 进一步确认。
Docker 在宿主机解析并冻结配置，容器不再次解析原始文件。`--server-artifacts` 关联启动目录时会一起复制这些记录。

4K 配置的总上下文是 4096 tokens，包含 P、S 和输出；GPU KV 池 8192 为预热后缀分支留余量。
首次 GPU 验收使用 `--prefix-lengths 2k --suffix-lengths 128 --groups cold l1 l2 --repeats 3 --gpu-timing`，结果目录单独新建。
当前提供的是配置和 CPU 回归覆盖，GPU 实测结果需在可用机器上另外记录。

## 已有环境：直接运行

在准备好 SGLang 依赖的环境中：

```bash
MODEL_DIR=/workspace/models/Qwen3-8B SGLANG_DIR=/workspace/sglang \
  HICACHE=1 FORWARD_TIMING=1 RUN_ID=server-demo \
  bash serve.sh
```

不设置 `GPU` 时继承当前可见设备；云平台已经只分配一张卡时通常不需要再设置。
普通服务器可用 `GPU=2` 选择设备，用 `PYTHON_BIN=/path/to/venv/bin/python` 选择解释器。
通过 `PYTHONPATH` 加载源码只替换 SGLang Python 代码，不安装它所需的 Torch、Kernel 等依赖。

启动记录和日志保存在 `RESULTS_DIR/RUN_ID`，默认项目的 `results/<UTC时间>`。
同名目录拒绝覆盖。`serve.sh` 最后 exec SGLang 进程，保留其退出码和信号行为。
已安装包模式记录模块位置和包版本；没有可识别的 Git checkout 时明确标记源码版本不可用。

## 需要 Docker 环境

宿主机需要 Docker、NVIDIA runtime、本地模型和本地已有镜像。可以先拉取此前使用的固定镜像：

```bash
export IMAGE=lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae
docker pull "$IMAGE"
MODEL_DIR=/path/to/Qwen3-8B SGLANG_DIR=/path/to/sglang GPU=0 \
  HICACHE=1 FORWARD_TIMING=1 RUN_ID=server-demo \
  CONTAINER_NAME=hicache-demo bash launch_server.sh
```

容器内 `/model`、`/experiment`、可选的 `/sgl-workspace/sglang` 只读；`/results` 可写。
Docker 入口不拼装模型参数；旧模式传递环境变量，命名模式传递冻结的配置，均调用 `serve.sh --exec`。
无配置名时的默认值仍在 `serve.sh` 定义；额外参数通过 Bash 数组传递，JSON 和带空格的路径保持完整。
`GPU` 只用于 Docker 设备选择，不把宿主机编号再次传为容器 CUDA 编号。

启动记录由宿主机写入，兼容 NFS root-squash。Docker 内运行 Driver 时仍识别 `EXPERIMENT_LAUNCH_DIR`。
退出时按本次创建的容器 ID 清理；容器退出码会传给启动脚本的调用者。镜像拉取或启动失败不会自动重试。

## 远程客户端

当前 `/debug/hicache` 只允许服务端本机连接。保留服务绑定 `127.0.0.1`，通过 SSH 转发端口：

```bash
# 替换 user@server 和 SSH_PORT；本地 39082 转到服务端 39081。
ssh -N -L 39082:127.0.0.1:39081 -p SSH_PORT user@server
```

另一个客户端终端运行：

```bash
python3 -m pip install -r requirements-client.txt
python3 run_experiment.py run \
  --base-url http://127.0.0.1:39082 \
  --tokenizer /local/path/to/Qwen3-8B-tokenizer \
  --output-dir results/remote-demo \
  --prefix-lengths 8k --suffix-lengths 128 \
  --groups cold l1 l2 --repeats 3 --gpu-timing
```

Qwen3-8B 客户端可复制模型的 `tokenizer.json`、`tokenizer_config.json`、`config.json`；无需复制权重。
其他模型应提供其完整 tokenizer 配套文件。脚本使用 `local_files_only=True`，不自动联网下载。
`client_latency_ms` 包含隧道/网络时间；比较服务端延迟和 GPU 区间时使用对应字段。

若需要完整来源证据，先把服务的启动目录复制到客户端，然后添加：

```bash
# 这里是客户端已经能读取的目录，不是远程路径。
# --server-artifacts /local/path/to/copied-server-demo
```

该参数可选；缺省仍采集 `/server_info`、`/model_info` 和请求原始响应，但不能据此确认具体镜像/源码。
`config.json` 的 `packages` 是客户端依赖版本；服务端版本来自可选启动记录，二者不可混淆。
启动记录是用户提供的来源证据，客户端不会自动验证该副本必然对应当前 URL。

## Vast.ai 普通容器

Vast 普通实例已经是容器，直接使用 `serve.sh`；不要在里面再次运行 `launch_server.sh`。
自定义模板可使用 SGLang 镜像，选择 SSH 模式。镜像应固定版本，并确认宿主驱动兼容。
官方文档：[容器环境](https://docs.vast.ai/guides/instances/docker-environment)、[模板设置](https://docs.vast.ai/guides/templates/template-settings)。

迁移顺序：准备实例 → 复制当前实验仓库和配套 SGLang checkout → 准备模型 → 验证 CUDA → 启动服务 → 小点/最大点验收 → 正式矩阵。
当前本地提交需要复制或先推送到自己的远程仓库，直接克隆官方 SGLang 不会获得本地实验补丁。
下层服务脚本不负责下载模型；需要自动准备源码、模型和整轮运行时使用 [run_all.py](cloud-task.md)。
租机、推理依赖安装、停止计费实例与回传数据仍由部署侧负责。

Qwen3-8B、BF16、最大 P=120K/S=2048 的启动参数示例：

```bash
MODEL_DIR=/workspace/models/Qwen3-8B SGLANG_DIR=/workspace/sglang \
  HICACHE=1 FORWARD_TIMING=1 HICACHE_SIZE=32 \
  CONTEXT_LENGTH=135168 MAX_TOTAL_TOKENS=135168 MEM_FRACTION=0.85 \
  RUN_ID=l1-server bash serve.sh \
  --json-model-override-args '{"max_position_embeddings":135168}'
```

在同一实例的另一个终端执行 Driver，`--tokenizer /workspace/models/Qwen3-8B`。
先用最大点 `--prefix-lengths 120k --suffix-lengths 2048 --groups l1 --repeats 1 --gpu-timing` 验收，另建正式目录后运行：

```bash
python3 run_experiment.py run \
  --base-url http://127.0.0.1:39081 \
  --tokenizer /workspace/models/Qwen3-8B \
  --server-artifacts results/l1-server --output-dir results/l1-matrix \
  --prefix-lengths 1k 2k 4k 8k 16k 24k 32k 40k 48k 56k 64k 72k 80k 88k 96k 104k 112k 120k \
  --suffix-lengths 16 32 64 256 512 768 1024 1536 2048 \
  --groups l1 --output-length 1 --repeats 10 --gpu-timing \
  --note 'Qwen3-8B，未缩放 RoPE 的性能外推'
```

正常完成为 1620 个有效样本。HiCache 仍开启用于备份/空闲检查和计时，L1 测量要求 host 命中和 H2D 传输均为零。
支持单次连续矩阵扫描；不会从失败目录自动续跑，旧的一次性协调器也不能直接搬到新机器使用。
Vast 真机 GPU 验收仍需执行；CPU 模拟回归只验证部署边界和请求流程。

## 相比旧启动方式的变化

- `MODEL_DIR` 现在显式指定，不再默认绑定个人 `/state/partition/...` 路径。
- Docker 默认 `GPU=0`；native 默认保留环境的可见卡。
- 服务参数入口集中为 `serve.sh`；`launch_server.sh` 仍是原来的 Docker 入口。
- 客户端只快照自身；启动脚本快照在服务目录的 `launch-source/`，通过 `--server-artifacts` 可一起关联。
- `--profile` 必须指定 `--server-profile-dir`；trace 留在服务端，复制到测量目录 `traces/` 后分析。
- `--telemetry-dir` 必须同时声明 `--telemetry-same-host`。跨主机单调时钟不可比较，不能直接把远程采样拷贝过来关联请求窗口。

历史实验记录和原始 CSV 保持原样；这些变化适用于新的运行命令。
