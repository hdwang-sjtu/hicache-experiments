# Qwen3.5 Cold / L1 / L2

[入口](../README.md) · [任务运行](cloud-task.md) · [架构](architecture.md)

支持 TP=1/2/4、EP=1 或 EP=TP，PP=DP=1。相同服务比较 cold、L1、L2；--groups l1 只保留 L1 正式样本，cold reference 仍在采样之外验证输出。

## 默认随机权重实验

    python3 run_all.py qwen35-35b-matrix \
      --gpu 0,1 --tp 2 --ep 2 --mem-fraction 0.8 \
      --prefix-lengths 16k 24k 32k \
      --suffix-lengths 1024 2048 16384 \
      --groups l1 --repeats 3

容器内部使用默认 native；宿主机需要 Docker 时增加 --runtime docker --image lmsysorg/sglang:dev。可指定 --model-dir 使用已有配置，省略则下载小文件到 HF 默认缓存。

默认 dummy 保留 FP8 模型结构和 BF16 KV，权重在 GPU 初始化；不会下载 FP8 权重分片。round-robin 使用 SGLang 自带模拟路由，不保留模型原来的语义。真实权重与原始路由的对照：

    python3 run_all.py qwen35-35b-matrix \
      --load-format auto --routing original --prefill-graph disabled \
      --model-dir /state/partition/hdwang/models/Qwen3.5-35B-A3B-FP8 \
      --gpu 0,1 --tp 2 --ep 2 --groups l1

122B 选择 qwen35-122b-128k，再按实例配置 TP/EP、P/S、容量和显存比例。35B 验收不代表 122B 或其他 GPU 的容量已经通过。

## 精确 P 检查点

Qwen3.5 混合注意力需要同时恢复完整 P 的全注意力 KV 与 GDN 线性状态。项目补丁保留 extra_buffer 检查点，并验收备份、GPU 状态、CPU 状态及逐 rank 操作。

P 按 256 对齐，chunk 为 256 的正整数倍；服务实际 checkpoint 网格还会由 Driver 检查。L1/L2 的 S 不超过固定 chunk，保证一次测量 forward。上下文和实际 KV 容量至少为最大 P+S+1。

每个样本先清空缓存；L1/L2 单独准备 P，再发送相同的 P+S。L2 在 CPU 备份完成后定向逐出 GPU，验收 KV 与 GDN 的实际恢复量。

## Prefill CUDA Graph

默认 breakable，decode Graph 关闭。L1/L2 自动使用去重 S；正式 cold 还加入 P+S 的 chunk 长度。不同 P 通常复用同一 token 桶，长 P 通过实时注意力元数据提供。

当前固定基线的 GDN 线性注意力在分段 Graph 中仍为 eager 边界，不能把模式名理解为整个 forward 完全捕获。所有 rank 必须在正式请求中报告预期 Graph 次数，否则任务失败。

round-robin 按本次 forward 的 token 序号分配专家，包含层与分片偏移。当前对齐设置满足该模型的轮转周期；调整切分方式时仍需检查 cold/L1/L2 输出一致，不能仅检查 token 数。

## 结果和证据

prefill_forward_gpu_ms 是最慢 rank 的原生 extend CUDA Event 区间，可包含通信、等待和提交间隙，不是所有 kernel 忙碌时间之和。

新结果记录 load_format、routing、prefill_graph 和 Graph/eager 次数。dummy 与真实权重、不同路由和不同 Graph 策略不能无条件合并拟合。旧数据缺少这些标记时保持 unknown。

历史实际配置、故障与性能见 [实验索引](experiments/README.md)。新模式验收仅证明记录中的硬件、模型、拓扑和配置组合。

## 从临时 L1 项目吸收的经验

以下链接固定到历史提交；其中旧入口和镜像内 SGLang 仅用于解释当时的证据，不是当前项目的运行命令。原始结果继续保存在 `hicache-experiments-vast-l1/results/native-l1/`，本次不搬迁、不改写。

- [35B TP2 验收](https://github.com/hdwang-sjtu/hicache-experiments-vast-l1/blob/8bd1fd289aa53de41a4d174d1482c45700c3d88d/docs/experiments/2026-11-04-qwen35-native-l1.md)：逐 rank 检查计算量、缓存命中和输出对照。
- [EP 与容量移植](https://github.com/hdwang-sjtu/hicache-experiments-vast-l1/blob/8bd1fd289aa53de41a4d174d1482c45700c3d88d/docs/experiments/2026-11-04-ep-capacity-port.md)：EP 标签和实际 KV 容量需要同时核对。
- [122B 未分块失败](https://github.com/hdwang-sjtu/hicache-experiments-vast-l1/blob/8bd1fd289aa53de41a4d174d1482c45700c3d88d/docs/experiments/2026-11-05-qwen122b-tp4-ep4-128k.md)：模型成功加载并不证明最大输入能运行。
- [显存比例扫描](https://github.com/hdwang-sjtu/hicache-experiments-vast-l1/blob/8bd1fd289aa53de41a4d174d1482c45700c3d88d/docs/experiments/2026-11-05-qwen122b-memfrac-sweep.md)：单独降低静态显存比例可能先耗尽 KV 容量。
- [32K/64K 分块验收](https://github.com/hdwang-sjtu/hicache-experiments-vast-l1/blob/8bd1fd289aa53de41a4d174d1482c45700c3d88d/docs/experiments/2026-11-05-qwen122b-chunked-prefill.md)：允许准备和 Cold 分块，同时保留 L1 后缀的一次 forward 验证。
