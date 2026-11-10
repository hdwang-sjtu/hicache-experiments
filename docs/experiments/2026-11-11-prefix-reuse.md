# 2026-11-11：跨 S 复用 P 的 TP4/EP4 验收

[实验索引](README.md) · [采样协议](../usage.md)

本轮检查同一个 P 只准备一次，所有 S 的预热和重复请求只读取 P、不把新增状态提交到共享缓存。cold/reference 仍完整计算 P+S，并在准备共享 P 之前集中完成。

## 环境与配置

- 四张 NVIDIA RTX 6000 Ada Generation，各 49140 MiB，驱动 570.124.06；使用 GPU 0–3。
- GPU 同时有其他用户任务。本轮验证功能、状态和实际请求数；耗时不作为独占 GPU 的性能基准。
- 模型 Qwen/Qwen3.5-35B-A3B-FP8，revision 9d1823d2dee688a6b25e77009dc727688c44936e；使用本地目录的小文件，dummy 初始化权重，round-robin 路由。
- TP4/EP4、PP=DP=1，BF16 KV，单请求、输出一个 token；context/max-total-tokens=51200，chunk=32768，mem-fraction=0.4，HiCache=2 GiB/rank。
- 镜像 lmsysorg/sglang:dev，实际 ID sha256:a1dc7080e3aa146f5d6ef1744f973b0591d8760d694f19badd9adce6f3fc6303。
- SGLang 基点 baa9befa381511cd4d56f83980951e5a45fffc6d，实验补丁 SHA256 为 4ff819038b2dad87c66cc577e34ac84c7216fc88002f88435042e981473159d2。

验收运行时项目尚未提交，以各结果目录的 code/ 和 run.json.code_sha256 为准确实现记录，最终提交不冒充当时已存在的版本。

## 已完成的验证

结果目录均位于项目 results/ 下；原始数据被 Git 忽略。

| 目录 | 配置 | 实际结果 |
|---|---|---|
| prefix-reuse-35b-tp4-native-v2-20261111 | native，P=2K/8K，S=64/1K，cold/L1/L2 各三次 | 36/36 有效；prepare 共两次，每个 P 一次 |
| prefix-reuse-35b-tp4-docker-boundary-20261111 | Docker，P=32K、S=16K，L1 一次 | 1/1 有效；prepare 一次 |
| prefix-reuse-35b-tp4-fresh-20261111 | native，关闭复用，P=8K，S=64/1K，L1 各三次 | 6/6 有效；包含各点预热，共 prepare 八次 |
| prefix-reuse-35b-tp4-eager-20261111 | native，复用开启、Graph 关闭，P=8K，S=64/1K，L1 各三次 | 6/6 有效；prepare 一次，各 rank 每请求 eager=1、Graph=0 |

四轮任务均为 complete，共 49 个正式有效样本。用最终总控验收函数重新读取各包，准备次数、有效样本数及逐 rank 计时/Graph 差值均通过。

主矩阵每个 P 的 cold 段完成后才开始命中段，各 S 的预热和重复均共用 P。正式请求每个 rank 都报告一次 Graph replay、零 eager；参考和准备请求允许 eager。Docker 边界点只捕获 16384 桶，完整 cold reference 分块执行。

复用请求前检查 P 的 GPU KV、GDN 状态和 CPU 备份；请求后检查完整 P+S 只能匹配到 P，并把所有 rank 的 KV/GDN 槽位与该 P 的初始基线比较。相同 P/S 的输出 token 与 cold 参考一致。

关闭复用的对照实际记录了 8 次 P=8K 准备；开启复用的主矩阵 P=8K 段只准备一次，并覆盖了两种 S、预热和 L1/L2 重复。这证明减少了准备请求，不能据此直接宣称端到端提速八倍。

主矩阵整卡 NVML 采样峰值为 17113 MiB，边界点为 19761 MiB，包含其他进程的占用，不能当成本服务精确显存峰值。

首次 native 尝试 prefix-reuse-35b-tp4-native-20261111 在参数解析阶段失败：总控把 HiCache 的 2 转成 2.0，而固定基线要求整数。已修复 CLI/环境覆盖类型并加入回归，失败包保留；它没有产生正式样本。

## 复现

在已有 SGLang 依赖的环境中，主矩阵命令为：

```bash
python3 run_all.py qwen35-35b-matrix \
  --runtime native \
  --model-dir /state/partition/hdwang/models/Qwen3.5-35B-A3B-FP8 \
  --gpu 0,1,2,3 --tp 4 --ep 4 \
  --prefix-lengths 2k 8k --suffix-lengths 64 1k \
  --groups cold l1 l2 --repeats 3 \
  --context-length 51200 --max-total-tokens 51200 \
  --chunked-prefill-size 32768 --mem-fraction 0.4 --hicache-size 2 \
  --run-id <新的运行名>
```

本机 native 验收在同一镜像创建的普通容器内执行以上入口，代码只读挂载、结果持久挂载。Docker 边界点由宿主机执行 run_all.py --runtime docker --image lmsysorg/sglang:dev，改为 P=32K、S=16K、groups=l1、repeats=1，其余参数相同。

关闭复用用 --no-prefix-reuse；关闭 Graph 用 --prefill-graph disabled。核对 prepare 次数可从 samples/responses.jsonl 筛选 path=/generate、stage=prepare，再按 response.meta_info.prompt_tokens 计数。timing_init、cold reference 和 warmup 另有阶段标记，不计入 prepare 次数。

## 验证边界

- 54 项 CPU 回归通过，包括跨 S/重复、cold 分段、L1/L2 切换、单 rank 泄漏、后缀污染、旧服务缺少能力及实际 prepare 次数被篡改的失败情况。Bash 语法、Markdown 链接/锚点/代码块和 Git whitespace 检查通过。
- 本轮服务与容器结束后清理，临时开发工作树删除，源码缓存与结果包保留。
- 本轮为 dummy 和模拟路由的功能验收，没有重跑 122B 或原始模型路由的性能实验。
- 输出 token 一致不是逐层数值误差证明；Graph 计数表示 breakable runner replay，GDN 仍有 eager 边界。
- 复用改变了缓存提交与测量间隔。比较客户端延迟或拟合数据时，应保留 prefix_reuse 标记及共享 GPU 条件。
- 成功的完整 /metrics 响应仍只在内存解析，保存的逐 rank 证据和缓存状态支持复查，不归档轮询全文。
