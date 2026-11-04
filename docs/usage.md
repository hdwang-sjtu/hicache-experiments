# 运行与指标参考

[返回工程入口](../README.md) · [部署与远程客户端](deployment.md) · [逐轮实验记录](experiments/README.md)

这里记录可复用的操作和定义；具体参数组合、实际结果与故障经过放在逐轮记录中。

## 怎么读这些入口

`serve.sh` 使用命名 YAML 或旧环境变量模式：读取配置 → 组装 SGLang 命令 → 保存启动记录 → exec 服务进程。
`launch_server.sh` 负责 Docker 镜像、挂载和生命周期，在容器内调用 `serve.sh`。
常用服务参数保存在 `configs/serve/`，通过共用的 `serve_config.py` 解析；旧模式默认值仍在 `serve.sh` 的 `server_args`。
修改挂载时看 `launch_server.sh` 的 `docker_args`。
两者通过内部工具 `launch_metadata.py` 保存记录。已有 Python 环境直接用 `serve.sh`，不需要 Docker。

`run_experiment.py` 可以按下面的调用顺序阅读：

```text
main()                       解析 run / analyze / self-test 子命令
└── run()                    准备目录和输入，等待并检查服务，保存最终状态
    ├── make_cases()         构造精确 P、S，单独准备 P
    ├── check_server()       检查 TP/EP、单请求、上下文长度和 KV 容量
    ├── run_samples()        执行运行预热，收集每组所需的有效测量
    │   └── execute_sample() 清空缓存 → 按组准备 → 测量 → 校验
    │       ├── Client      发送 HTTP 请求并保存原始响应
    │       └── validate()  检查长度、各层命中、回撤和时间字段
    └── analyze()            过滤无效样本，输出 Median、P95 和图表
```

有两种容易混淆的“预热”：`warmup` 是每个长度组合开始时的运行预热，不写入测量 CSV；`prepare` 是每一次 L1 样本都必须重新做的 P 请求，用来构造缓存状态。
`measure` 才是计入 CSV 的 P+S 请求，且其清空和准备耗时不计入客户端测量延迟。

函数说明、关键流程和命令行帮助使用中文。变量、CSV 字段和图表标签保留英文；内置测试语料也保持原文，以免修改注释时改变输入。
可用 `bash serve.sh --help`、`bash launch_server.sh --help` 和 `python3 run_experiment.py run --help` 查看中文参数说明。

## L2：先在 8K 上验收

先在旧服务终端按 Ctrl-C 停止旧服务。SGLang 补丁需要重新启动进程才生效。
选择一张空闲且 CUDA 正常的 GPU 后，在终端 A 运行：

```bash
cd /path/to/hicache-experiments
export MODEL_DIR=/path/to/Qwen3-8B
GPU=3 HICACHE=1 RUN_ID=l2-demo CONTAINER_NAME=hicache-l2 bash launch_server.sh
```

`HICACHE=1` 固定 Python TreeCore，开启 CPU 持久缓存、`write_through` 和本地实验接口；CPU KV 池默认 8 GB（十进制，约 7.45 GiB），可用 `HICACHE_SIZE` 修改。没有启用 L3。
三组必须在这个相同服务上重新测量，不能把旧版关闭 HiCache 的 Cold/L1 直接混进来。

服务就绪后，在终端 B **先执行真实树/allocator 的定向逐出测试**：

```bash
docker exec --workdir /sgl-workspace/sglang/test/registered/unit/mem_cache hicache-l2 \
  python -m unittest test_unified_radix_cache_unittest.TestHiCacheExperiment -v
```

这组测试检查 P 边界拆分、后缀子节点、锁引用、备份未完成、重复逐出和额外分支，使用真实树及 device allocator；host 槽位管理被替代，真实 D2H/H2D 需由下一条实验验证。
测试通过后执行：

```bash
docker exec --user "$(id -u):$(id -g)" hicache-l2 \
  python /experiment/run_experiment.py run \
  --tokenizer /model --output-dir /results/l2-demo-samples \
  --prefix-lengths 8k --suffix-lengths 128 --output-length 1 \
  --groups cold l1 l2 --repeats 3
```

完成后应有三组各 `valid 3/3`。通过后才能改为 `--prefix-lengths 8k 16k 32k --repeats 20`，同时选择新的 `--output-dir`，例如 `/results/l2-scan`。

### 受控缓存操作

实验开关默认关闭。开启后，Driver 调用 `POST /debug/hicache`，发送 `action=inspect/demote` 和 `prefix_ids`。
HTTP 进程只转发消息，操作在 Scheduler 中执行：

JSON/IPC 传入的 token IDs 是 `list`；接口在匹配前转换为 serving 使用的 `array('q')`。
`RadixKey` 本身不做这种转换，树内键与查询键的容器类型必须一致。缓存回归测试也使用真实 serving 的数组键。

- `inspect` 等待请求和异步操作全部结束，返回 Prefix 的 GPU 命中、CPU 备份覆盖及两级池的已用/可用槽位。
- `demote` 要求整个目标分支的备份完成、无锁引用、无额外共享分支；统一预检后，从后缀叶节点向根复用现有逐出机制。
- 操作保留 host KV 和索引；GPU allocator 可用槽位增加，但静态 KV 池不会因此从 `nvidia-smi` 消失。
- 不支持的组件、Rust TreeCore、额外分支或不完整备份都会报错，不能把失败降级成普通请求继续测量。

每个样本的准备如下，均在测量区间外：

| 组别 | 准备操作 | 测量请求的实际命中要求 |
|---|---|---|
| Cold | 清空 → 确认空闲 | device=0，host=0 |
| L1 | 清空 → P → 等待 CPU 备份完成 | device=P，host=0 |
| L2 | 清空 → P → 等待备份 → Demote | device=0，host=P |

注意：L1 的 CPU **备份覆盖**也等于 P，但请求实际 **host 命中**为 0，因为 GPU 已有副本。脚本分别记录这两个量。

## 参数与结果

可使用 [命名服务配置](deployment.md#命名服务配置)，也可继续通过旧环境变量模式启动，额外参数直接追加给 SGLang：

```bash
GPU=1 MODEL_DIR=/path/to/local/model DTYPE=bfloat16 KV_DTYPE=bfloat16 \
  RUN_ID=another-model bash launch_server.sh
```

可覆盖 `SGLANG_DIR`、`IMAGE`、`PORT`、`CONTEXT_LENGTH`、`MAX_TOTAL_TOKENS`、`MEM_FRACTION`、`RESULTS_DIR`。
`TP`/`EP` 是启动参数；Driver 支持 TP=1/2/4、EP=1/TP，要求 PP=DP=1。多 rank 必须启用所有 scheduler 指标与 GPU timing。
无配置名时默认上下文 40960、KV 容量上限 36864 tokens、chunk size 4096、并发 1、Graph 关闭、overlap 保留。
`qwen3-8b-4k` 则使用 4096 总上下文和 8192 KV 槽位，应使用 P=2K 的小点验收，不能照搬本页旧模式的 8K/32K 命令。
其他模型需自行设置正确上下文长度，dtype 必须由模型及所用后端支持。

## GPU 计时和 Profiler

本 checkout 已有 `DeviceTimer`：在当前 forward stream 上用 CUDA Event 包围 `model.forward`，
通过非阻塞 `query()` 检查完成，再更新 `sglang:forward_execution_seconds_total{category="extend"}`。
这里直接启用和读取它；本阶段没有增加 SGLang 计时补丁。

重新启动专用服务，三组都启用相同的计时设置：

```bash
MODEL_DIR=/path/to/Qwen3-8B GPU=1 HICACHE=1 FORWARD_TIMING=1 RUN_ID=timing-demo \
  CONTAINER_NAME=hicache-timing bash launch_server.sh
```

第二个终端先采集诊断时间线，缓存准备与逐出不进入 Profiler 区间：

```bash
docker exec --user "$(id -u):$(id -g)" hicache-timing \
  python /experiment/run_experiment.py run \
  --tokenizer /model --output-dir /results/profile-demo \
  --prefix-lengths 32k --suffix-lengths 128 --output-length 1 \
  --groups cold l1 l2 --repeats 1 --gpu-timing --profile \
  --server-profile-dir /tmp/hicache-profile-demo --no-plot

# 将服务端本机磁盘的 trace 归档到本轮结果，再进行分析。
docker cp hicache-timing:/tmp/hicache-profile-demo results/profile-demo/traces

docker exec --user "$(id -u):$(id -g)" hicache-timing \
  python /experiment/run_experiment.py analyze-traces /results/profile-demo
```

`--profile` 只开启测量请求期间的原生 `/start_profile` / `/stop_profile`，预热、备份和 demote 在区间外。
它有额外开销，诊断样本只用于解释时间线，不能混入关闭 Profiler 的性能结果。
`--server-profile-dir` 是服务端绝对路径，服务端需有写入权限，Driver 不创建或修改该远程目录。
NFS root-squash 等情况下应事先为服务用户准备可写目录；可先在服务端本机磁盘采集，结束后再复制。
远程运行时，将服务端 trace 复制到客户端本轮目录的 `traces/`，再执行 `analyze-traces`。
为避免 trace 与样本归属歧义，`analyze-traces` 要求每个 P/S/组恰好一次有效诊断测量。

输出 `profile_summary.csv/json` 和 `profile_timeline.png`：

- `forward_kernel_busy_ms`：通过 CPU `step[EXTEND ...]` 与 CUDA correlation 归属到 forward 的 Kernel 区间并集；包含 Kernel 自身的访存时间，排除区间之间的空隙。
- `h2d_kernel_busy_ms`：当前 kernel 后端的逐层 KV 恢复 Kernel 区间并集；不包含 D2H 备份和普通输入的小型 H2D memcpy。
- `overlap_ms`：上述两组区间的交集；不是用两个延迟相减估算。
- `h2d_without_forward_kernel_ms`：H2D Kernel 在执行、forward Kernel 没在执行的时间；不能直接解释为可优化掉的请求延迟。

H2D Kernel 名称依据本 checkout 的 `pool_host/mha.py` 和 HiCache Kernel 源码识别。
其他传输后端或新的 Kernel 名称需扩展识别规则；识别不到 L2 加载时会报错，不会输出假零值。
forward 计时不包含此前的 attention metadata 准备；Profiler 的 step 范围稍宽，因此两个值不能直接相减求等待时间。

验证时间线后，关闭 Profiler，正式扫描 S：

```bash
docker exec --user "$(id -u):$(id -g)" hicache-timing \
  python /experiment/run_experiment.py run \
  --tokenizer /model --output-dir /results/timing-scan-demo \
  --prefix-lengths 32k --suffix-lengths 32 128 512 2048 --output-length 1 \
  --groups cold l1 l2 --repeats 20 --gpu-timing
```

固定 P 的 `latency.png` 横轴为 S、延迟使用对数轴；`timing.png` 对比 L1 forward、L2 H2D 和 L2 forward 的 Median/P95。
图中的流区间可以重叠，不做串行堆叠。原来的 8K/16K/32K 结果继续保留，不能与不同 GPU 或计时设置的新结果混合计算分位数。

## 环境监控

为区分缓存正确与环境可观测，Driver 提供可选 `--telemetry-dir`。
该功能要求 Driver 与采样进程位于同一主机、共用单调时钟；SSH 隧道连接不代表同一主机。
先按下文准备采样目录和 `owner.json`，再在宿主机的两个终端分别启动采样（`TELEMETRY_DIR` 替换为实际宿主机路径）：

```bash
python3 run_experiment.py monitor --kind gpu --output-dir "$TELEMETRY_DIR" --interval 1
python3 run_experiment.py monitor --kind host --output-dir "$TELEMETRY_DIR" --interval 1
```

使用监控前，先创建 `owner.json`，填写选定物理 GPU 编号、专用容器允许的 PID，以及 Scheduler PID。需要在宿主机用 `docker top <container> -eo pid,args` 核对进程归属，不应把其他任务加入允许清单。
GPU 采样使用 NVML，记录四卡的计算进程、利用率、显存、SM/显存时钟、温度、功率和限频原因。
各卡的 NVML 错误单独保存；GPU 采样的整体 `ok` 表示选定实验卡是否可观测。
其他卡报错不会丢弃实验卡的数据，选定卡故障仍会暂停；整次查询超过 3 秒的检查继续保留。
主机采样独立记录各核 CPU、负载、内存、磁盘 I/O、压力指标及 Scheduler CPU/排队统计；NVML 查询阻塞不会停止主机心跳。
传感器不支持时保留缺失原因，不写成假零。两种采样都记录单调时钟、UTC 时间和查询耗时。

样本的测量起止时刻与两路采样关联：前后边界距离不超过 2 秒，窗口内相邻采样间隔不超过 3 秒，查询不能失败或持续超过 3 秒。
选定 GPU 出现额外计算进程或监控失效时，下一次请求在计时外暂停；5 分钟不恢复则停止并保存进度，不操作其他人的任务。
未解释的慢样本仍然保留，不能按延迟阈值删掉以让曲线变直。

- `cache_valid`：缓存命中、长度、传输计数、正常结束及输出一致性检查。
- `environment_valid`：上述采样覆盖及选定 GPU 的进程检查。
- `valid`：两项同时满足，才计入每组的 10 个正式样本。
- `summary.csv`：正式有效样本的统计；`all_cache_valid_summary.csv`：所有缓存有效尝试的统计，包括环境检查失败者。

这种采样仍不能证明整机无干扰；其他卡的活动和共享主机资源可能产生影响，原始监控与异常理由随数据保存。
`prefill_forward_gpu_ms` 仍是流区间，可能包含提交间隙，不能叫作纯 Kernel 时间。


`owner.json` 格式示例（PID 必须替换为当前实验服务的真实宿主机 PID）：

```json
{"gpu_index": 3, "allowed_gpu_pids": [12345], "scheduler_pid": 12345}
```

GPU 采样需要宿主机的 `pynvml`，主机采样需要 `psutil`。监控目录必须先存在且可写。
随后运行 Driver 时添加 `--telemetry-dir /results/<采样目录> --telemetry-same-host`；路径需在客户端可读。
云容器中的 NVML PID、进程命名空间与可见主机范围需要现场核对，不能把缺失可观测性当成无干扰。
两个监控应在正式请求前启动，使用目录中的 `stop` 文件结束采样；目录和 PID 清单每次新建。
Driver 本身不负责后台作业编排。`run_all.py` 管理整轮服务和测试，但尚不自动启动遥测或续跑正式样本。
历史逐轮协调器源码保存在相应结果的 `audit/`，其中路径/PID 不可直接复用。


## 样本如何构造和判定

默认把脚本内固定的英文实验笔记分词后重复、截取到精确长度。这是合成性能输入，不用于评价模型回答质量。
可以用 `--text-file /path/to/text.txt` 指定固定 UTF-8 文本；特殊 token 不允许进入测试文本。

每个 P/S 组合先对所选组各执行一次运行预热，不计入 CSV。每次测量分别执行：

1. Cold：确认缓存清空成功 → 发送 P+S。
2. L1：确认缓存清空成功 → 发送 P 生成 1 token → 发送相同测量输入 P+S。

准备请求只处理 P。每次重复都清空后重新准备，避免上次的 S 也被缓存。
每轮随机化组顺序（种子可复现），单请求串行执行。

有效样本必须满足：

- 输入 tokens = P+S，输出 tokens = 指定值，正常到达长度限制；temperature=0、ignore_eos=true。
- Cold 总命中 / device / host / storage 均为 0。
- L1 总命中和 device 恰好为 P，host / storage 为 0。
- L2 总命中和 host 恰好为 P，device / storage 为 0；实际 KV H2D tokens=P；字节数还需包括混合模型的线性状态，逐 rank 校验后合计。
- 回撤次数为 0；服务端必须返回完整、有效的时间统计。
- 相同 P/S 的有效测量应生成相同 token IDs；L2 模式要求开启 `--return-output-ids`（启动器已设置）。

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
| `kv_load_ms` | 请求前后 `load_back_duration_seconds_sum` 差值转成 ms，表示已完成 H2D 操作的 GPU-stream 区间累计值 |
| `kv_load_tokens` / `kv_load_bytes` | 逻辑 KV H2D token 数 / 所有 rank 的 KV 与线性状态恢复字节合计 |
| `kv_load_ops` | 本次请求完成并带有 CUDA Event 时间的 H2D 操作数 |
| `prefill_forward_gpu_ms` | 可选；原生 extend CUDA Event 累计指标的请求前后差值。Cold 累加全部 chunk；L1/L2 只计算 S，但 L2 区间可能等待 KV |
| `device_tokens_before` / `host_backed_tokens_before` | 准备结束后，目标 P 在 GPU 上的命中 / CPU 上的备份覆盖 |
| `device_used_tokens_before/after` / `host_used_tokens_before/after` | 请求前后，缓存池中已用的 token 槽位数，不是 CUDA allocator 占用字节数 |

HiCache 模式下，接口先确认空闲，脚本再读取 `/metrics` 基准；测量请求结束后等 ACK 被处理，再读取终值。两个指标快照之间没有 flush，也不包含其他测量请求。
GPU timing 任务逐 rank 读取 KV/Mamba 池与 forward 指标；时间取最大 rank，逻辑 token 数不累加，字节数合计。未启用 GPU timing 的旧 FULL-KV 单卡路径只读取 KV 指标。首次加载前显式初始化零值；缺失、重复、倒退或传输量不符时，不得作为有效样本。
H2D 时间包含逐层提交之间的空隙，但不含之前的排队、索引准备和 fence 等待；它可与 Prefill 重叠，不能直接与首 token 延迟相加。
关闭 HiCache 的旧结果没有这些测量，新列留空；`analyze` 继续兼容原 CSV。

Median 和 P95 仅对有效样本计算；P95 使用排序后线性插值。少量重复用于流程验证；有限样本的 P95 不能据此推断稳定的生产尾延迟。
图只绘制已达到目标有效次数的组，同时标记整个运行状态。

`/flush_cache` 在当前 SGLang 中也会调用 device allocator 的 `empty_cache()`。
因此本阶段 Cold/L1 差异可能包含瞬态分配状态差异，是缓存基线检查；不能把差值当作纯计算量，也不能外推为 KV 加载时间。

## 原始结果文件

每次运行保存到 `results/<RUN_ID>/`：

- `launch.json`、可选 `sglang.patch`、`gpu.txt`、`server.log`：保存在启动目录；用 `--server-artifacts` 可关联其客户端副本。native 无镜像 ID，未识别 Git checkout 时无源码提交号。
- 命名模式额外保存 `profile.yaml`、`server-config.yaml`、`profile-resolution.json`，记录原文、环境覆盖及 CLI；`launch.json` 保存配置名和哈希。
- `config.json`：Driver 参数、状态、实际 `/server_info`、模型信息、客户端依赖版本、tokenizer 哈希。
- `inputs.json`：固定文本、精确 prefix / suffix / 预热 suffix IDs 和测量输入哈希。
- `responses.jsonl`：控制操作、预热和测量的原始响应，阶段分别标记。
- `samples.csv`：每个测量样本及有效性；`summary.csv`：有效样本 Median / P95 和无效计数。
- `latency.png`：服务端首 token 延迟图（需要 matplotlib）；`source/` 保存 Driver 运行版本；`launch-source/` 保存可选的服务启动脚本快照。

目录不得复用。失败尝试保留原始数据并标明状态，不计为已完成实验。
一个服务可对应多个测量目录，后续目录关联原始启动信息；没有启动证据时无法证明镜像和源码版本。

混合注意力、多 rank 的状态与指标规则见 [Qwen3.5 实验](qwen35.md)。旧记录中的 P+S₀ 方法属于当时的 FULL-KV 实验，新入口使用独立 P 准备。
