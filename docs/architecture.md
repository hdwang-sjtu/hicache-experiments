# 项目架构与调用关系

[工程入口与调用图](../README.md) · [一键云任务](cloud-task.md) · [分步部署](deployment.md) · [实验用法与指标](usage.md) · [逐轮实验记录](experiments/README.md)

本工程负责实验编排、服务启动、HTTP 采样和结果校验；模型计算与 KV 缓存操作在配套 SGLang 中执行。
源码 checkout 和模型权重通过路径接入，可由总控准备到缓存，也可由部署侧提前准备。

## 入口与调用顺序

完整任务从 `run_all.py <实验名>` 开始，调用图见 [README](../README.md)：

1. 读取实验配置和版本锁，复用 `serve_config.py` 解析服务配置，复用 Driver 的参数解析检查实验设置，包括总上下文、HiCache 与 GPU 计时的依赖。创建本轮快照、取得端口锁并检查端口空闲。
2. 准备运行环境：Docker 模式拉取或复用锁定镜像，native 模式检查当前 Python 依赖。再分别启动 `prepare_runtime.py source` 和 `prepare_runtime.py model`，获取并校验固定源码及模型。
3. 按 `--runtime` 启动服务：native 调用 `serve.sh`；Docker 调用 `launch_server.sh`，由它在容器内调用 `serve.sh --exec`。最终都是 `python -m sglang.launch_server`，`serve.sh` 以 `exec` 交接服务进程。
4. 总控通过 `/health` 等待就绪，再用 `/server_info` 的 `served_model_name` 核对本轮服务身份，随后单独启动 Driver。native 使用当前环境，Docker 使用 `docker exec` 在本轮容器内运行 Driver；两者均通过 HTTP 请求服务。
5. Driver 保存响应、有效性校验、统计和图表；总控进一步检查完成状态、必要文件及各组合的有效样本数。成功、失败或中断都进入清理，停止本轮进程组或按已记录的容器 ID 删除本轮容器，最后写入任务状态。

`serve.sh`、`launch_server.sh` 和 `run_experiment.py` 都可以绕过总控独立使用。
手工启动服务后，用户另开终端运行 Driver，也可以从远程客户端经 [SSH 隧道](deployment.md#远程客户端) 请求服务。
服务进程不负责启动 Driver；总控也不管理云实例租用、销毁或停止计费。

## 文件职责与调用者

### 执行入口与共享工具

| 文件 | 调用者 | 职责 |
|---|---|---|
| [run_all.py](../run_all.py) | 用户或平台启动命令 | 编排准备、服务、Driver、结果验收与清理；保存任务状态。导入 Driver 复用参数解析，实际采样仍启动独立进程。 |
| [prepare_runtime.py](../prepare_runtime.py) | 总控的准备子进程；总控也导入其锁与 JSON 写入工具 | 固定基线检出、补丁校验与应用、固定 revision 模型下载和缓存完整性检查；准备阶段与推理计时分开。 |
| [serve.sh](../serve.sh) | 用户、总控 native 分支、Docker 包装 | 在当前 Python 环境组装并执行 SGLang 命令；支持命名配置和旧环境变量模式。旧模式服务默认参数集中在此处。 |
| [launch_server.sh](../launch_server.sh) | 用户或总控 Docker 分支 | 将本地镜像引用固定为镜像 ID，设置 GPU、挂载和容器生命周期；调用容器内 `serve.sh`。独立使用时要求镜像已在本地。 |
| [run_experiment.py](../run_experiment.py) | 用户或总控；总控还导入其参数解析函数 | `run` 构造精确 token 输入、控制缓存、串行请求、校验和汇总；`analyze` 重算统计与图表；另提供 `self-test`、`monitor`、`analyze-traces`。可单独复制到客户端。 |
| [serve_config.py](../serve_config.py) | 总控及两个启动脚本 | 读取命名 YAML，解析路径与显式覆盖，输出启动配置和预览。只处理配置数据，不导入 SGLang/Torch；完整服务参数由 SGLang 原生解析器检查。 |
| [launch_metadata.py](../launch_metadata.py) | 两个启动脚本 | 写入启动命令、源码版本/差异、配置、启动器快照及 GPU 信息；Docker 记录镜像身份，native 记录当前 Python 包信息。使用标准库，不参与请求和推理。 |

### 配置、依赖与补丁

| 文件 | 读取方 | 职责 |
|---|---|---|
| [configs/experiments/qwen3-8b-smoke.yaml](../configs/experiments/qwen3-8b-smoke.yaml) | `run_all.py` | 一轮实验的模板：引用服务配置名，指定 Hugging Face 模型仓库和完整 revision，保存 Driver 的 `test` 参数。 |
| [configs/serve/qwen3-8b-4k.yaml](../configs/serve/qwen3-8b-4k.yaml) | `serve_config.py`，供总控和两个启动脚本使用 | 服务模板：相对模型目录名、运行时环境变量及 SGLang 参数。可迁移的服务设置放在这里，GPU、镜像和本机绝对路径由启动环境提供。 |
| [configs/runtime.lock.json](../configs/runtime.lock.json) | 总控；其中源码相关字段由准备工具使用 | 固定镜像摘要、SGLang 仓库与基线、补丁路径和 SHA256，另记录补丁来源提交。 |
| [patches/sglang-hicache-experiment.patch](../patches/sglang-hicache-experiment.patch) | 总控校验并快照，`prepare_runtime.py source` 应用 | 随工程分发的 SGLang 实验接口及缓存 inspect/demote 实现与回归测试；需要显式开启实验开关。 |
| [requirements-launch.txt](../requirements-launch.txt) | 部署侧通过 pip 安装 | 命名配置解析和总控模型下载依赖；不安装推理环境。 |
| [requirements-client.txt](../requirements-client.txt) | 客户端通过 pip 安装 | 本地 tokenizer 与绘图依赖；客户端不需要 SGLang、模型权重、Docker 或 GPU。 |

### 回归与文档

| 文件或目录 | 使用者 | 职责 |
|---|---|---|
| [tests/test_deployment.py](../tests/test_deployment.py) | unittest | 用模拟服务、解释器和 Docker 检查启动记录、配置冻结、路径重写、独立 HTTP 客户端及远程路径/时钟边界。 |
| [tests/test_run_all.py](../tests/test_run_all.py) | unittest | 检查总控成功/失败清理、已有端口保护、精确容器身份及模型补齐与复用。 |
| [README.md](../README.md) | 使用者 | 工程入口、简图和最小运行命令。 |
| [docs/architecture.md](architecture.md) | 维护者 | 文件职责、调用关系、配置与数据边界。 |
| [docs/cloud-task.md](cloud-task.md) | 任务部署者 | 总控用法、自动准备、缓存、状态及清理规则。 |
| [docs/deployment.md](deployment.md) | 服务部署者 | 命名服务配置、native/Docker 分步启动和远程访问。 |
| [docs/usage.md](usage.md) | 实验执行者 | 缓存准备、指标定义、Profiler、遥测和分析操作。 |
| [docs/experiments/](experiments/README.md) | 实验执行者与复核者 | 逐轮记录实际配置、样本、结论、故障及其原始证据。 |

## 两个启动脚本怎样共用工具

命名模式下，两种入口都使用 `serve_config.py`，遵循 **命令行 > 显式环境变量 > 配置文件**。
native 由 `serve.sh` 解析配置，再调用 `launch_metadata.py` 写记录，最后执行 SGLang。
Docker 由宿主机上的 `launch_server.sh` 解析模型路径并将容器路径改为 `/model`，调用同一记录工具保存配置，再将冻结的 `profile-resolution.json` 交给容器内的 `serve.sh --exec --resolved-profile`。
容器内仅解包已解析配置，不重新读取原始 YAML，也不重复生成启动记录。

命名启动保存的 `profile.yaml` 是原文，`server-config.yaml` 是环境覆盖后的下发配置，`profile-resolution.json` 保存路径、环境和 CLI 信息；CLI 仍可覆盖下发 YAML。
最终配置再由 Driver 获取的 `/server_info` 确认。具体覆盖和模型路径规则见 [命名服务配置](deployment.md#命名服务配置)。

不带配置名时，旧环境变量模式继续可用；默认服务参数由 `serve.sh` 统一定义。
Docker 包装只转交显式变量和追加参数，并调用 `serve.sh --exec`；启动记录仍由宿主机调用 `launch_metadata.py` 写入。
两个启动脚本都不下载模型或准备 SGLang 源码，这些工作由总控或部署侧完成。

## 实验配置、服务配置与版本锁

实验名与服务名分别在 `configs/experiments/` 和 `configs/serve/` 中查找。总控通过实验 YAML 的 `serve` 字段关联服务模板：

- **实验配置**决定测什么：P/S、组别、重复次数等 Driver 参数，以及模型仓库和固定 revision。地址、tokenizer、结果路径由总控生成。
- **服务配置**决定怎样提供推理服务：上下文、KV 容量、精度、HiCache 和计时等；它的顶层 `model` 是本地目录名，不等于模型仓库或模型版本。
- **版本锁与补丁**决定总控准备的运行代码：`sglang_base` 是实际 checkout 的 HEAD，工作区再应用锁定哈希的补丁；`patch_source_commit` 记录补丁来源，并不是准备时检出的 HEAD。官方镜像本身不会自动包含本地实验补丁。

总控将固定模型放入 `<MODEL_ROOT>/<服务配置的 model>/<revision>/`，并将准备好的源码缓存路径传给 `SGLANG_DIR`。
源码缓存复用前检查基线与实际差异；模型缓存通过版本标记和文件清单检查完整性，缺失或变化时补齐/重新下载。详细规则见 [固定版本与缓存](cloud-task.md#固定版本与缓存)。

版本锁由总控使用，独立启动脚本不会自动读取它或应用补丁。
手工部署时，未设置 `SGLANG_DIR` 会优先使用存在的相邻 `../sglang`，也可显式指定 checkout；`SGLANG_DIR=''` 使用当前环境或镜像中的安装包。
这些代码来源仍须与实验接口和指标兼容。通过 `PYTHONPATH` 接入源码只替换 Python 代码，不会安装 Torch、CUDA 或 Kernel 依赖。
Docker 总控会使用锁定镜像并记录实际镜像 ID；native 总控保留期望镜像与实际软件环境，不能据此认定所在环境的镜像摘要已验证。

## Driver 与服务的 HTTP 边界

| 接口 | Driver 的用途 |
|---|---|
| `/health`、`/server_info`、`/model_info` | 等待就绪，读取并检查真实服务设置和模型信息。总控也使用前两个接口核对服务身份。 |
| `/flush_cache`、`/debug/hicache` | 在测量区间外清空缓存、检查备份/空闲状态，或定向逐出 GPU 副本以构造 L2 起点。 |
| `/generate` | 用 token IDs 发送准备请求和正式测量请求，读取输出与服务端指标。 |
| `/metrics` | 读取测量前后计数器，计算 H2D 传输和 GPU forward 区间差值。 |
| `/start_profile`、`/stop_profile` | 独立 Driver 按需采集诊断 trace；不属于总控正式采样配置。 |

`/debug/hicache` 来自实验补丁，需显式开启且只接受服务端 loopback 请求。
HTTP 层转发请求，实际操作经 TokenizerManager 消息交给 Scheduler 和缓存实现，详见 [受控缓存操作](usage.md#受控缓存操作)。
Driver 不直接调用服务端 Python 对象，也不要求共享磁盘：`--tokenizer` 和 `--output-dir` 是客户端路径，`--server-profile-dir` 是服务端 trace 路径。
可选的 `--server-artifacts` 接收客户端可读的启动目录或副本，用来关联来源证据；该副本本身不能证明对应当前 URL。
统计与绘图读取已保存的 CSV，`analyze` 不需要运行中的服务。

## 结果目录与缓存目录

`results/` 保存每轮独立证据，`.cache/` 保存可复用的准备产物；二者均由 [.gitignore](../.gitignore) 排除。
克隆仓库不会带回原始数据或模型，结果保留与同步由部署侧负责。

| 位置 | 内容与用途 |
|---|---|
| `results/<run-id>/`（总控） | `run.json` 记录任务状态与失败阶段；实验 YAML、版本锁、补丁和 `code/` 保存本轮快照；`source.json`、`model.json` 保存准备来源；各阶段日志用于排障。完整布局见 [结果、失败与清理](cloud-task.md#结果失败与清理)。 |
| 总控的 `server/`；独立启动的 `RESULTS_DIR/RUN_ID` | `launch.json`、配置解析记录、`launch-source/`、源码差异、GPU 信息和服务日志；Docker 另保存 `container.id`。 |
| 总控的 `samples/`；独立 Driver 的 `--output-dir` | 输入、原始响应、样本 CSV、汇总、图表及 Driver 快照；可关联启动记录。字段与统计规则见 [运行与指标参考](usage.md)。 |
| `.cache/sglang/<基线>-<补丁哈希>/checkout/` | 总控校验后复用的源码，避免修改手工准备的 checkout。 |
| `.cache/models/<model>/<revision>/` | 默认版本化模型目录；`HICACHE_CACHE_DIR` 可改缓存根目录，`MODEL_ROOT` 可单独改模型根目录。 |
| `.cache/locks/` 及源码/模型缓存旁的锁文件 | 限制同一端口的任务和同一缓存的并发准备。 |

`RESULTS_DIR` 可修改结果根目录。总控和启动器拒绝复用已有运行目录，Driver 拒绝覆盖已有测量；重新执行使用新目录，准备缓存可复用。
[.gitattributes](../.gitattributes) 为补丁设置 whitespace 例外，以保留统一 diff 的上下文空行字节；修改补丁内容后须同步版本锁中的哈希。

## 验证边界

历史 GPU 实验验证的是当轮归档的 Driver、SGLang 和配置组合；不能将后续入口重构视为已经在相同硬件上验收。
具体样本数、结果和故障经过以 [实验记录索引](experiments/README.md) 及各轮记录为准，准确版本以当轮启动元数据、补丁和源码快照为准。

当前 CPU 回归使用模拟 HTTP 服务、Docker 和下载器，覆盖启动记录、配置覆盖、客户端请求、总控编排和失败清理；它不证明 CUDA、真实缓存搬运或 GPU 性能。
新入口与新平台仍需在目标机器分别完成小点、容量边界及实际 GPU/平台验收，操作见 [分步部署](deployment.md) 和 [一键云任务](cloud-task.md)。
