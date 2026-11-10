# 2026-11-10：dummy / round-robin / Prefill Graph 验收

[实验索引](README.md) · [当前使用方法](../qwen35.md)

本轮验收脚本重构后的流程，不作为新的性能基准。运行时工具代码尚未提交，准确版本以每个结果包的 code/ 和 run.json.code_sha256 为准；最终整理提交不能代替当轮快照。

## 环境和共同配置

- GPU：同一节点的 0、1 卡，NVIDIA RTX 6000 Ada Generation，各 49140 MiB；驱动 570.124.06。
- 镜像：lmsysorg/sglang:dev，本地 ID sha256:a1dc7080e3aa146f5d6ef1744f973b0591d8760d694f19badd9adce6f3fc6303。
- SGLang：固定 baa9befa381511cd4d56f83980951e5a45fffc6d，加版本锁中的实验补丁 45913465ca6b28a7c01c2a3dd1a186adcc52367728c9d2d691af3fc712a20d0f。
- 模型配置：Qwen/Qwen3.5-35B-A3B-FP8，revision 9d1823d2dee688a6b25e77009dc727688c44936e；权重由 dummy 初始化，路由 round-robin。
- TP2/EP2，BF16 KV，单请求，输出一个 token，context/max-total-tokens=51200，chunk=32768，mem-fraction=0.8。

三个 Docker 总控任务使用已有模型目录，但 load-format=dummy，不读取权重。另用只含八个配置/tokenizer/processor 文件的临时目录验证了 native 流程：该目录完全没有权重或分片索引，model.json 记录 weight_bytes=0。

## 实际结果

| 任务结果目录（项目 results/ 下） | 正式网格 | 有效样本 | 所有 rank 的正式执行路径 |
|---|---|---:|---|
| refactor-35b-eager-20261110 | P=2K，S=1024/2048，cold/L1/L2，各重复两次 | 12/12 | 每请求 Graph=0、eager=1 |
| refactor-35b-graph-20261110 | 同上 | 12/12 | 每请求 Graph=1、eager=0 |
| refactor-35b-boundary-20261110 | P=32K、S=16K，L1，一次 | 1/1 | Graph=1、eager=0 |
| refactor-35b-native-assets-20261110 | P=2K、S=1024，cold/L1/L2，各一次 | 3/3 | 每请求 Graph=1、eager=0 |

四轮均为 complete，共 28 个正式有效样本。缓存命中、KV/GDN 起点、计算 token、forward 次数、L2 恢复量、输出 token 一致性及总控二次验收通过。reference/prepare/warmup 不计入上表。

Graph 小矩阵自动捕获 1024/2048/3072/4096：前两项是命中后的 S，后两项是 cold 的 P+S。边界点只捕获 16384，准备 P 和 cold reference 可以 eager；正式 L1 仍必须 Graph。

小矩阵四个桶的捕获耗时约 56 秒，启动日志报告每 rank Graph 显存约 0.58 GB。NVML 采样峰值：eager 20921 MiB/卡，Graph 小矩阵 21627 MiB/卡，边界点 25961 MiB/卡；采样峰值不是分配器精确峰值。

每个验收结果目录约 0.8–1.2 MB；两组 S 的 inputs.json 为 1638 bytes。成功 /metrics 轮询全文没有归档，仅保存选定的逐 rank 证据；完整服务日志只有 server/server.log 一份。此大小只描述本轮小实验。

## 复现入口

在项目根目录运行；源码缓存路径按机器调整：

    HICACHE_CACHE_DIR=/tmp/hicache-unified-20261105/data/cache \
    python3 run_all.py qwen35-35b-matrix \
      --runtime docker --image lmsysorg/sglang:dev \
      --model-dir /state/partition/hdwang/models/Qwen3.5-35B-A3B-FP8 \
      --gpu 0,1 --tp 2 --ep 2 \
      --prefix-lengths 2k --suffix-lengths 1024 2048 \
      --groups cold l1 l2 --repeats 2 \
      --context-length 51200 --max-total-tokens 51200 \
      --chunked-prefill-size 32768 --mem-fraction 0.8 \
      --prefill-graph breakable --run-id <新的运行名>

eager 使用 --prefill-graph disabled。边界点改为 P=32K、S=16K、groups=l1、repeats=1。native 验收在相同镜像创建的普通容器中执行 run_all.py --runtime native，模型挂载仅含小文件，其余共同设置相同；结果保存在宿主机同一 results/ 根目录。

## 回归和限制

- 46 项 CPU 回归覆盖 native/Docker 冻结启动、移机入口、精简远程客户端、dummy 下载白名单、输入重建、多 rank 指标延迟与 Graph 回退。
- 旧双卡 756 行、四卡 1008 行数据在内存中重算；2520 项 median/p95 与原 summary.csv 一致，没有修改历史数据。
- Markdown 本地链接、锚点、代码块、Bash 语法和 Git whitespace 检查通过。
- 本轮没有重跑 122B、完整 P/S 矩阵或真实权重性能对照。少量重复仅用于流程验收；生成 token 一致不等于逐层数值误差验证。
- breakable Graph 中 GDN 线性注意力仍走 eager 边界，Graph 次数表示 runner replay，不能解释为全部 kernel 均在一个图内。
- 四轮结束后均清理本轮资源，GPU 0、1 已恢复空闲。原始结果被 Git 忽略，clone 不包含这些数据。
