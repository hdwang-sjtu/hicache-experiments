# DeepSeek V4 Flash：双 H200 cold/L1

[入口](../README.md) · [协议与指标](usage.md)

目标模型是 `deepseek-ai/DeepSeek-V4-Flash`，版本固定在实验 YAML。保留原版 FP4 专家与 FP8 dense/attention 结构，使用 dummy 权重，不下载权重分片。该模型配置不适用于 V4.1 或其他 V4 变体。

**双 H200 NVL 已通过 eager 和 breakable CUDA Graph 的 cold/L1 小网格。** TP=2、EP=1、dummy 权重，每种模式各得到 24 条有效样本。这里的容量验收覆盖下文的小网格，128K 长输入另测。

## 从小网格开始

进入已有 SGLang 依赖的容器，在项目根目录执行；`native` 表示使用当前容器的 Python 环境：

```bash
python3 -m pip install -r requirements-launch.txt

# 先加 --dry-run 可查看最终参数，不下载、不启动、不写结果目录。
python3 run_all.py deepseek-v4-flash \
  --runtime native --gpu 0,1 \
  --prefill-graph disabled \
  --run-id dsv4-h200-smoke-eager

# eager 验收通过后，单独测试 Graph，避免把两种模式混入一个运行目录。
python3 run_all.py deepseek-v4-flash \
  --runtime native --gpu 0,1 \
  --prefill-graph breakable \
  --run-id dsv4-h200-smoke-graph
```

每轮默认 TP=2、EP=1、单并发，P 为 2K/4K，S 为 128/1024，cold/L1 各三次有效测量，另有两轮预热。两轮共 48 个正式样本；初始化、预热和 P 准备不计入其中。关闭推测解码、CP、DP attention 和 decode Graph。

H200 使用 FlashInfer MXFP4 W4A16 路径：FP4 专家权重、BF16 激活、FP8 KV。`swa-full-tokens-ratio: 1.0` 是首版保守页预算，避免显式限制 KV 容量后，SWA 配额小于一次 chunk；它不表示把滑动窗口改成全注意力。需要宿主机创建 Docker 时改用 `--runtime docker`，默认镜像和 SGLang 源码仍由版本锁固定。

服务在 H200 加载权重前检查 FlashMLA，优先使用固定镜像的 `sgl_kernel.flash_mla`。当运行环境把它移到独立 `flash_mla` 包时，使用该包的 legacy 内核，沿用原版 V4 的 KV 布局。所选路径记录在 `server/server.log`。首次启动还会预编译 DeepGEMM，等服务就绪后才开始采样。

## 扩大 P/S 网格

沿用现有参数即可；P 按 256 对齐，L1 的 S 不超过固定 chunk（默认 32K）。上下文和逻辑容量默认按最大 P+S+1 向上调整；显式指定不足的容量会报错。实际各池容量仍须由服务端验收。

```bash
python3 run_all.py deepseek-v4-flash \
  --gpu 0,1 --groups l1 \
  --prefix-lengths 16k 24k 32k \
  --suffix-lengths 1024 2048 16384 \
  --repeats 7 --run-id dsv4-h200-ps-001
```

先用单点检查长输入容量；这不是性能统计：

```bash
python3 run_all.py deepseek-v4-flash \
  --gpu 0,1 --prefill-graph disabled \
  --prefix-lengths 128k --suffix-lengths 16k \
  --groups cold l1 --repeats 1 --run-id dsv4-h200-capacity-eager
```

成功后再以新的 run-id 和 `--prefill-graph breakable` 检查 Graph。OOM、命中或 Graph 校验失败均保留失败记录；不会自动降低输入长度或切换精度后宣布原配置成功。

## 测量与实现边界

- 仅支持 **cold/L1**，关闭 HiCache；请求 L2、启用 HiCache 或 EP>1 会在启动前拒绝。当前没有 CPU 备份、卸载或恢复实验。
- 复用现有 `/debug/hicache` 的 `inspect`，在所有 rank 空闲后调用正常 FULL+SWA 匹配，不能用 FULL 历史完整代替 SWA 完整。此模式拒绝 `demote`，也拒绝未验收的 request-ring/replay KV 布局。
- 同一 P 只准备一次；每次 P+S 读取 P，禁止发布 S。C4 KV 和 indexer 跟随 FULL 页，C4 压缩状态跟随 SWA 页，C128 的请求状态跟随请求槽。每次结束都对照 P 的初始基线检查独立分配器，且完整 P+S 仍只能命中 P。
- dummy 初始化覆盖原生 E8M0 缩放的打包 INT8/UINT8 FP4 权重，直接填充有限的随机 FP4 编码及小幅值缩放，不临时展开整套 BF16 权重。沿用 SGLang dummy 的固定初始化种子；真实权重加载不经过此步骤。
- round-robin 同时覆盖普通 TopK 和前三层 HashTopK 的路由专家，保留共享专家及缩放约定。256 对齐的 P/chunk 满足该模型的轮转周期；模拟路由下的时间不能直接代表真实路由的时间。
- Graph 桶由测量长度生成。正式 cold 的所有 chunk、L1 的 S 都逐 rank 检查执行路径。Graph 配置名不表示所有注意力内部 kernel 均被完整捕获。

结果仍在 `results/<run-id>/`：`server/server.log` 看服务，`driver.log` 看进度，`samples/` 保存结果。`samples/config.json` 记录实际缓存布局，`rank-metrics.jsonl` 保存逐 rank 证据。没有 CPU 恢复时 H2D 指标列留空，不能解释为测得了零耗时恢复。`prefill_forward_gpu_ms` 为最慢 rank 的 forward CUDA Event 区间，可能包含通信和等待。

小网格的 GPU 验收使用 dummy 权重，覆盖所测配置的计算与缓存行为。真实权重和更长输入的容量需要各自实测。
