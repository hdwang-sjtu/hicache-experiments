# GPU 计时与固定 P 扫描 S（2026-10-31）

## 目标和配置

区分请求延迟、已完成 H2D 流区间和 forward 流区间，用独立 Profiler 诊断验证加载与计算重叠。
固定 P=32K、S=32/128/512/2048，Cold/L1/L2 各 20 次，共 240 个正式有效样本、0 个无效样本。
正式扫描时间为 18:33–19:03（Asia/Shanghai）；Profiler 样本单独保存，不计入正式性能统计。

服务启用 `HICACHE=1 FORWARD_TIMING=1`；Driver 添加 `--gpu-timing`，诊断请求另外添加 `--profile`。
启动设置见 `timing-scan/launch.json`，复用命令和计时定义见[GPU 计时和 Profiler](../usage.md#gpu-计时和-profiler)。

## 结果和证据

本机 [timing-scan](../../results/timing-scan/) 的中位数如下，单位 ms；全部指标的 Median/P95 见 `summary.csv`。
配置为 Qwen3-8B、GPU 1、TP=1、BF16 权重/KV、P=32768、输出 1 token，Profiler 关闭。

| S | Cold 首 token | L1 首 token | L2 首 token | L1 forward 区间 | L2 H2D 区间 | L2 forward 区间 |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 6485.42 | 30.70 | 192.42 | 26.46 | 187.10 | 186.83 |
| 128 | 6492.16 | 39.22 | 189.77 | 34.70 | 184.20 | 184.04 |
| 512 | 6601.06 | 120.90 | 193.23 | 116.20 | 185.64 | 187.01 |
| 2048 | 7014.02 | 520.88 | 545.46 | 513.72 | 258.08 | 534.99 |

每个 L2 样本均为 device=0、host=32768，实际恢复 4,831,838,208 字节（4.5 GiB）KV。
短 S 下恢复时间占主导；S=2048 时计算区间变长，L1/L2 延迟接近。固定 P 时 H2D 区间也会随并发执行变化，不能预设为常数。
`validation.json` 记录了对原始响应、原始指标差值、逐出状态、槽位变化、输出一致性及全部分位数的复核。

诊断时间线分别位于 [timing-profile-v3](../../results/timing-profile-v3/)（S=128）和 [timing-profile-s2048](../../results/timing-profile-s2048/)，
每个目录都是 Cold/L1/L2 各一次有效诊断请求，不进入上面的正式统计。
下表仅列这两个 L2 诊断请求，单位 ms：

| S | forward Kernel 忙碌区间 | H2D Kernel 忙碌区间 | H2D Kernel 首尾跨度 | 两者重叠 | H2D 忙碌区间的重叠比例 |
|---|---:|---:|---:|---:|---:|
| 128 | 41.23 | 188.14 | 188.23 | 38.95 | 20.7% |
| 2048 | 502.31 | 189.19 | 244.58 | 182.79 | 96.6% |

两次都观察到 36 次逐层 KV 加载。S=2048 时 H2D Kernel 首尾跨度明显大于忙碌区间，
说明流上的时间包含加载 Kernel 之间的间隙；不能全部解释为传输 Kernel 的执行时间。
大部分加载执行区间与计算重叠，与正式测量中 L1/L2 延迟接近的现象一致。
每组的 `triage-*.txt` 保存仓库 Profiler 分析技能生成的 Kernel、Overlap、Fuse 三张表；
这里没有据此提出 Kernel 融合或调度优化。

## 源码版本与限制

SGLang 基点为 `baa9befa381511cd4d56f83980951e5a45fffc6d`，加未提交的本地缓存控制补丁；GPU 计时直接复用原生指标，没有更改模型计算或新增计时补丁。
正式扫描 Driver SHA256 为 `347bb5bbfc82083edf3cde318b3ca811a662594a04b82430245459167aa5c137`；后续 S=2048 Profiler 分析使用的 Driver SHA256 为 `1c2df937c88390faf402d7349f72c1a7df14f9d0ea9a1b351f33bb60b32932c7`。
准确源码以各自 `source/` 和 `sglang.patch` 为准，不能把独立诊断和正式扫描视为完全相同的 Driver 版本。
计时表示本轮观测；该轮未采用后来矩阵的连续双路环境有效性检查，缓存验证通过不构成整机无干扰证明。
