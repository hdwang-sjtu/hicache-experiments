# Cold/L1 基线与 L2 验收（2026-10-30 至 10-31）

## 目标和配置

先确认 GPU Prefix 复用，再验证历史 KV 仅在 CPU 时能恢复到 GPU。模型为 Qwen3-8B，BF16 权重/KV、TP=1、并发 1、输出 1 token，S=128。
基线关闭 HiCache；L2 验收三组共用开启 HiCache 的服务。两种服务配置分别统计。

## 运行记录

| 目录 | 配置 | 状态与样本 |
|---|---|---|
| [baseline-8k](../../results/baseline-8k/) | GPU0，P=8K，Cold/L1 各 3 次 | complete，6 个有效样本；流程验证 |
| [baseline-scan](../../results/baseline-scan/) | GPU0，P=8K/16K/32K，Cold/L1 各 20 次 | complete，120 个有效样本 |
| [l2-8k](../../results/l2-8k/) | GPU0，P=8K，三组各 3 次 | failed，0 个测量样本 |
| [l2-8k-fix1](../../results/l2-8k-fix1/) | 相同 P=8K 小规模验收 | complete，9 个有效样本 |
| [l2-scan](../../results/l2-scan/) | GPU0，P=8K/16K/32K，三组各 20 次 | complete，180 个有效样本 |

基线正式扫描时间为 10 月 30 日 15:29–15:37；L2 正式扫描为 10 月 31 日 01:30–01:41，均为 Asia/Shanghai。
具体参数和启动命令见各目录的 `config.json`、`launch.json`；[通用 L2 操作](../usage.md#l2先在-8k-上验收)保留可复用命令。

## 验证和结论

`l2-scan/validation.json` 记录五项缓存树/allocator 测试通过，以及对全部原始响应、准确命中、实际 H2D 字节、前后槽位、输出 ID 和分位数的复核。
五项测试使用真实树和 device allocator，host 槽位管理被替代；真实 D2H/H2D 由服务请求验证。
L2 请求满足 device=0、host=P，说明历史 KV 从 CPU 恢复，而非 GPU L1 命中。GPU 静态 KV 池的显存仍由 SGLang 持有，不能用 `nvidia-smi` 显存下降判断 demote 是否成功。
该阶段验证缓存流程和基本延迟，尚未单独导出 GPU forward 计时，不能把客户端请求时间当作 Prefill 或 H2D 时间。

## 失败经过

最初 `l2-8k` 在 `/debug/hicache` 返回 HTTP 500。HTTP 输入为 list，而 serving 树键使用 `array('q')`，匹配时容器类型不一致触发断言。
接口增加规范化转换后，`l2-8k-fix1` 与完整 L2 扫描通过；失败目录不进入统计。

## 源码版本

运行时 SGLang 基点为 `baa9befa381511cd4d56f83980951e5a45fffc6d`；基线无本地补丁，L2 使用未提交的控制补丁。
历史 Driver SHA256：`baseline-scan` 为 `11b5801c52e6675f8ba6440aceab06e2106458936494d21cce8078a7100f41ed`；`l2-scan` 为 `be8cd1d35d8aa9fed5dad94c798a712e1be8178b22bfbfbba4653c51f64e436a`。
这些运行发生在后续代码提交之前，以各目录 `source/` 和 `sglang.patch` 为运行源码证据，不能追认成新提交版本。整理后的提交映射见[索引](README.md#源码版本与提交检查点)。
