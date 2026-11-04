# 实例内一键实验任务

[返回工程入口](../README.md) · [服务配置与分步部署](deployment.md)

`run_all.py` 是给 Vast.ai 等平台的无交互任务入口：准备源码和模型 → 启动服务 → 等待健康检查 → 执行 Driver → 验收结果 → 清理本次服务。
它运行在已准备好驱动与推理依赖的实例内，不调用云平台租机、停机或销毁 API。

## 用法

```bash
python3 -m pip install -r requirements-launch.txt
python3 run_all.py --list-experiments
python3 run_all.py qwen3-8b-smoke --dry-run
python3 run_all.py qwen3-8b-smoke --run-id smoke-001
```

默认 `--runtime native`，适用于 SGLang 官方镜像提供的环境、Vast 普通容器或已配置的服务器。
宿主机仅有 Docker 时：

```bash
GPU=0 python3 run_all.py qwen3-8b-smoke --runtime docker --run-id smoke-docker-001
```

Docker 模式自动拉取/复用版本锁中的镜像，Driver 也在该容器内运行，宿主机只需 Python、Git、配置解析和下载依赖。
两种模式都沿用 `GPU` 和服务参数环境变量；总控自动管理 `SGLANG_DIR`、模型路径、服务身份和容器名。
`--dry-run` 只检查配置与显示任务，不联网、不启动、不创建目录。`--run-id` 缺省为 UTC 时间加实验名；同名目录拒绝覆盖。

## 实验配置

配置位于 `configs/experiments/`，服务参数继续放在 `configs/serve/`。示例：

```yaml
serve: qwen3-8b-4k
model:
  repo: Qwen/Qwen3-8B
  revision: b968826d9c46dd6066d109eabc6255188de91218
test:
  prefix-lengths: [2k]
  suffix-lengths: [128]
  groups: [cold, l1, l2]
  output-length: 1
  repeats: 3
  gpu-timing: true
```

这轮目标是 9 个有效样本。4K 是服务总上下文，P+S+输出必须容纳其中。
`test` 沿用 Driver 参数名，支持长度/组别/重复次数、预热、种子、超时、GPU 计时、说明及 `no-plot`。
`text-file` 的相对路径以实验 YAML 所在目录为基准，内容会复制进本轮任务。
服务地址、tokenizer、输出和启动记录路径由总控产生。Profiler 和遥测暂时使用独立 Driver，避免混入正式性能任务。

## 固定版本与缓存

`configs/runtime.lock.json` 固定官方镜像摘要、SGLang 基线、仓库内补丁 SHA256。
源码准备先检出指定基线，再执行 `git apply --check` 和应用补丁；同时验证实际差异哈希。
它不修改你已有的 SGLang checkout。缓存中的源码如果被改动会报错，不会自动重置用户文件或重复打补丁。
当前补丁来源由版本锁记录，提供 FULL/FULL+MAMBA 的备份检查、多 rank 协同驱逐和稳定状态检查，不修改模型计算。

模型使用 Hugging Face 固定 commit 下载，缺少分片或中断下载时会补齐。
完成后保存版本与文件清单；后续只有同仓库、同版本且文件完整的缓存才直接复用。仅有 `config.json` 不算完整模型。
已完成缓存发生文件变化时重新校验/下载，不把旧完成标记当成有效证据。

默认布局：

```text
.cache/
├── sglang/<基线>-<补丁哈希>/checkout/
├── models/<服务配置的模型目录名>/<模型revision>/
└── locks/
```

可设置 `HICACHE_CACHE_DIR` 修改缓存根目录，`MODEL_ROOT` 单独修改版本化模型缓存根目录。
`--model-dir` 或 `MODEL_DIR` 可只读使用现有本地模型，检查分片索引和 safetensors 头部，不做全权重哈希，也不写入完成标记。
固定 revision 缓存和手工模型分别记录来源。`--source-repository` 可指定含固定基线的本地 Git 仓库作为离线源码来源，补丁校验仍执行。
下载阶段与推理计时完全分开。认证由环境中的 Hugging Face 凭据提供，不写入实验 YAML 或任务状态。

Docker 模式默认使用版本锁中的镜像，允许 `--image` 覆盖，并记录其本地镜像 ID。native 模式只记录“期望镜像”和真实软件环境，不能从容器内部假装已验证镜像摘要。
脚本不安装或升级 Torch/CUDA；依赖不匹配、GPU 不可用等问题会作为启动失败留下日志。

## 结果、失败与清理

```text
results/<run-id>/
├── run.json                  # 总体状态、阶段、退出码及失败原因
├── experiment.yaml           # 原始实验配置
├── resolved-experiment.json  # 应用 CLI 覆盖后的实验配置
├── gpu-memory.jsonl          # 启动至清理期间的采样显存与阶段
├── runtime.lock.json         # 版本锁
├── sglang.patch              # 本轮补丁
├── code/                     # 实际使用的入口/Driver/服务配置快照
├── source.json / model.json  # 准备阶段来源与复用信息
├── prepare.log               # 依赖检查、源码及模型准备日志
├── server-launch.log         # 服务启动器日志
├── driver.log                # 逐组进度和 Driver 错误
├── server/                   # SGLang 启动证据、配置、日志、容器 ID
└── samples/                  # 输入、响应、CSV、Median/P95、图表
```

状态为 `running`、`complete`、`failed` 或 `interrupted`。失败时 `failed_phase` 指明准备、启动、测试、结果验收或清理阶段。
源码准备/拉取镜像默认最多 30 分钟，模型下载最多 2 小时；服务就绪超时沿用 Driver 的 `ready-timeout`，默认 600 秒。
服务提前退出会立即失败，不必等满就绪时间。测量中服务退出也会终止测试。

开始前确认端口空闲，就绪后用原生 `served-model-name` 核对本任务的随机身份；不向其他服务发送清缓存或测试请求。
完成要求 Driver 正常退出、状态 complete、必要输出文件存在，且每个 P/S/组的有效样本数与配置一致；GPU timing 任务还重新核对逐 rank 原始指标与 CSV。
不启用 `no-plot` 时还要求相应延迟/GPU 计时图生成。

成功、失败和 SIGINT/SIGTERM 都清理本任务创建的服务：native 按独立进程组，Docker 按已记录的容器 ID。
结果保留到指定目录；成功返回 0，失败非零，中断返回 130/143。
任务完成状态在清理后写入。退出实验脚本不代表云实例已经停机或停止计费。

## 配置到 Vast 等平台

平台实例使用版本锁指定的官方 SGLang 镜像。实验仓库本身需要先上传、挂载，或从你发布的仓库拉取一次。
无需自建实验镜像，也无需另行维护 SGLang fork：源码补丁已随本工程分发。

工程已放在实例内时，启动命令可以是：

```bash
cd /workspace/hicache-experiments
export HICACHE_CACHE_DIR=/workspace/cache
export RESULTS_DIR=/workspace/results
exec python3 run_all.py qwen3-8b-smoke --run-id smoke-001
```

若由平台在全新实例上拉取工程，可先执行以下准备命令，再使用上面的启动命令：

```bash
# 在平台环境变量里填写实际仓库地址和本次固定提交，不写入 Git 凭据。
git clone "$EXPERIMENT_REPO_URL" /workspace/hicache-experiments
git -C /workspace/hicache-experiments checkout --detach "$EXPERIMENT_COMMIT"
python3 -m pip install -r /workspace/hicache-experiments/requirements-launch.txt
```

Vast 的 SSH/Jupyter 模式会替换原镜像 entrypoint，应将任务命令放到 On-start 配置；具体入口见 [Vast 模板设置](https://docs.vast.ai/guides/templates/template-settings)。
结果目录应指向需要保留的磁盘或挂载卷；实例销毁前的结果取回由部署侧负责。

不包括正式样本断点续跑、完整环境有效性遥测、结果上传和云实例计费回收。任务自带的显存采样用于诊断，不代替环境有效性判断。重新执行使用新任务目录；源码/模型缓存可复用。
CPU 回归只验证流程和故障清理，真实 GPU 性能与 Vast 部署需要在目标机器上验收。

Qwen3.5 的 TP/EP、P/S、分块及容量覆盖命令见 [Qwen3.5 实验](qwen35.md)。Vast On-start 中也使用同一 `run_all.py` 命令，并把 `RESULTS_DIR` 指向持久卷。
