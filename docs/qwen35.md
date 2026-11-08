# Qwen3.5 Cold / L1 / L2

[工程入口](../README.md) · [统一任务](cloud-task.md) · [架构](architecture.md)

统一入口是 `run_all.py`，HTTP Driver 是 `run_experiment.py`。支持 TP=1/2/4，EP=1 或 EP=TP，PP=DP=1。三组使用同一套开启 HiCache 的服务；`--groups l1` 只选择正式采样组，仍会做 Cold 正确性对照。

## 本地模型与 Docker

在具备 SGLang 运行依赖的环境（包括已经启动的 SGLang 容器）中：

```bash
python3 run_all.py qwen35-35b-matrix \
  --model-dir /state/partition/hdwang/models/Qwen3.5-35B-A3B-FP8 \
  --tp 1 --ep 1 --gpu 0 --run-id qwen35-35b-001
```

仅有 Docker 的宿主机使用相同入口：

```bash
python3 -m pip install -r requirements-launch.txt
python3 run_all.py qwen35-35b-matrix --runtime docker \
  --image lmsysorg/sglang:dev \
  --model-dir /state/partition/hdwang/models/Qwen3.5-35B-A3B-FP8 \
  --tp 2 --ep 2 --gpu 0,1 --run-id qwen35-35b-tp2-001
```

Docker 的 `/model` 和源码只读，`/results` 挂载到宿主机 `RESULTS_DIR/<run-id>`。HF 快照挂载所属 HF 缓存根目录，服务和 Driver 均使用 `/model/models--<组织>--<模型>/snapshots/<revision>/`，保留仓库内及跨仓库共享 `blobs/` 的软链接。容器使用宿主 UID/GID。GPU 列表指宿主机编号，容器内不重复设置同一组物理编号。

`--model-dir` 或 `MODEL_DIR` 指定完整模型目录：完整可用就直接使用，目录不存在或缺文件时按实验 YAML 的仓库和固定 revision 补齐。检查配置、tokenizer、分片索引和 safetensors 头部，不计算全部权重哈希；结构损坏或已有来源明确冲突时报错。本地 HF metadata 或快照目录名仅作为 revision 线索，不冒充完整内容验证。
省略目录则交给 HF 库使用默认缓存，直接使用返回路径；通常位于 `~/.cache/huggingface/hub`，沿用 `HF_HOME` / `HF_HUB_CACHE`。总控不再使用 `MODEL_ROOT`。缓存和云平台挂载见 [一键任务](cloud-task.md#固定版本与缓存)。

`--image` 可覆盖版本锁中的默认镜像，Docker 任务记录实际 image ID。镜像提供 Torch/CUDA 等运行依赖；SGLang Python 源码仍来自固定基线加仓库补丁，不能仅凭镜像内置版本判断实验代码。

如果 Docker daemon 无法遍历 NFS 私有目录，先把项目、缓存和结果目录放到宿主机本地磁盘，再使用上述命令。完成后将结果复制回工作区。普通 Vast 容器直接使用 native 入口，不需要嵌套 Docker。

## 122B 与长前缀

```bash
export HICACHE_CACHE_DIR=/workspace/cache
export RESULTS_DIR=/workspace/results
python3 run_all.py qwen35-122b-128k \
  --model-dir /workspace/models/Qwen3.5-122B-A10B-FP8 \
  --tp 4 --ep 4 --mem-fraction 0.8 \
  --prefix-lengths 128k --suffix-lengths 2048 \
  --chunked-prefill-size 32k --repeats 3 --run-id qwen35-122b-001
```

在只有两张 PRO6000 的目标机器上，可以改成 `--tp 2 --ep 2`，但必须在那里重新验收容量和最大点。配置允许该组合，不代表本地四张 RTX 6000 Ada 的记录能证明两张 PRO6000 可运行。

先加 `--dry-run` 检查解析结果。默认上下文至少容纳最大 P+S+1，并按 2048 对齐；KV 配置至少容纳这个长度。启动后还检查服务报告和各 rank 的实际池容量。

默认 chunk 为 `min(context, 32768)`。Cold 和准备 P 可以分块，按预期次数检查 forward；L1/L2 的 S 必须不大于 chunk，以保持一次 forward。Qwen3.5 配置要求 P/chunk 按 256 对齐，启动后进一步检查模型的 checkpoint 网格。显式容量不够时直接报错，不静默缩短输入。

`--hicache-size` 默认每 rank 8 GB；TP4 对应四份主机池预算。容量和分块减少已知的失败因素，但不能保证任意模型、显存比例或平台没有 OOM。完整启动和请求期间的采样显存写入 `gpu-memory.jsonl`；它是采样峰值，不是 GPU 分配器的精确瞬时峰值。

## 为什么能保存 P 的线性状态

每次 L1/L2 都先清空，再单独发送 P，且只生成一个输出 token。该请求完成 P 的 prefill 后采样输出，不继续把这个输出送入下一次 forward。因此可在 P 的对齐边界保存 `H(P)`，同时保存对应全注意力层 KV。

服务使用 `mamba-radix-cache-strategy=extra_buffer`，后续 P+S 使用请求自己的状态副本，保留树上的 P 检查点。这里复用 SGLang 的检查点和复制机制，不把线性状态当成可以任意切片的 KV。

- Cold：清空后直接执行 P+S。
- L1：准备 P，确认所有 rank 的 GPU KV 和线性状态、CPU 备份均完整，再执行 P+S。
- L2：准备 P，所有 rank 先检查空闲、检查点、备份和锁；全部通过才驱逐 GPU 副本，再检查主机副本保留，然后执行 P+S。

检查失败时不会开始驱逐；执行阶段若部分 rank 失败，则返回致命实验错误，由总控停止所属服务，不继续测量。实验接口仍只接受本机连接。

## 结果与解释

`results/<run-id>` 保存实际配置、代码与补丁、来源、服务日志、样本、汇总和图。`samples/rank-metrics.jsonl` 保存准备、预热、测量请求在每个 rank 上的原始指标和差值。

- GPU forward 时间和 H2D 时间取各 rank 最大值，不把卡数相加成端到端时间。
- token 数是逻辑长度，不累加 TP rank；`kv_load_bytes` 为所有 rank 的实际恢复字节合计，混合模型包含 KV 和 Mamba 状态。
- L2 必须同时恢复 P 的 KV 与线性状态；树检查点和请求副本可能产生一份或两份状态传输，按实际池字节大小核算。
- Cold/L1 必须没有缓存 H2D 恢复；相同输入在三组中的输出 token 必须一致。所有计数和异步事件完整发布并稳定后才结算，等待过程不计入 HTTP 请求延迟。

历史实验、当前入口 GPU 验收及 CPU 模拟回归是不同证据。[本次整合记录](experiments/2026-11-05-qwen35-unified.md)记录实际验收环境和结果；历史 L1 经验见下方固定提交链接。`hicache-experiments-vast-l1` 工作区继续保留，可在主项目验收期间独立使用；它的旧结果不归入新入口的验收样本。

## 从临时 L1 项目吸收的经验

以下链接固定到历史提交；其中旧入口和镜像内 SGLang 仅用于解释当时的证据，不是当前项目的运行命令。原始结果继续保存在 `hicache-experiments-vast-l1/results/native-l1/`，本次不搬迁、不改写。

- [35B TP2 验收](https://github.com/hdwang-sjtu/hicache-experiments-vast-l1/blob/8bd1fd289aa53de41a4d174d1482c45700c3d88d/docs/experiments/2026-11-04-qwen35-native-l1.md)：逐 rank 检查计算量、缓存命中和输出对照。
- [EP 与容量移植](https://github.com/hdwang-sjtu/hicache-experiments-vast-l1/blob/8bd1fd289aa53de41a4d174d1482c45700c3d88d/docs/experiments/2026-11-04-ep-capacity-port.md)：EP 标签和实际 KV 容量需要同时核对。
- [122B 未分块失败](https://github.com/hdwang-sjtu/hicache-experiments-vast-l1/blob/8bd1fd289aa53de41a4d174d1482c45700c3d88d/docs/experiments/2026-11-05-qwen122b-tp4-ep4-128k.md)：模型成功加载并不证明最大输入能运行。
- [显存比例扫描](https://github.com/hdwang-sjtu/hicache-experiments-vast-l1/blob/8bd1fd289aa53de41a4d174d1482c45700c3d88d/docs/experiments/2026-11-05-qwen122b-memfrac-sweep.md)：单独降低静态显存比例可能先耗尽 KV 容量。
- [32K/64K 分块验收](https://github.com/hdwang-sjtu/hicache-experiments-vast-l1/blob/8bd1fd289aa53de41a4d174d1482c45700c3d88d/docs/experiments/2026-11-05-qwen122b-chunked-prefill.md)：允许准备和 Cold 分块，同时保留 L1 后缀的一次 forward 验证。
