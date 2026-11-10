# 实验记录索引

[返回工程入口](../../README.md) · [运行与指标参考](../usage.md)

这里保存随 Git 版本管理的实验记录。每轮写明目标、关键配置、源码版本、运行命令或配置入口、实际结果、结论及限制，并链接到原始数据。
失败和中断也应记录实际进度；planned、failed 与 complete 不能混写。具体排障证据保存在对应结果目录，不追加到工程 README。

## 已有记录

| 实验 | 状态 | 原始数据 |
|---|---|---|
| [2026-11-11：跨 S 复用 P 的 TP4/EP4 验收](2026-11-11-prefix-reuse.md) | native、Docker、长输入及开关对照共 49 个有效样本 | results/prefix-reuse-35b-tp4-*-20261111/ |
| [2026-11-10：dummy / round-robin / Prefill Graph 验收](2026-11-10-synthetic-prefill.md) | eager、Graph、容量点和无权重 native 共 28 个有效样本 | results/refactor-35b-*-20261110/ |
| [2026-11-05：Qwen3.5 统一入口 TP/EP 验收](2026-11-05-qwen35-unified.md) | 35B 五种组合共 180 个、122B 长前缀 9 个有效样本 | `results/unified-qwen35/` |
| [Cold/L1 基线与 L2 验收](baseline-and-l2.md) | 基线 120 个、L2 180 个正式有效样本；另有流程验证和失败尝试 | `results/baseline-*`、`results/l2-*` |
| [2026-10-31：GPU 计时与固定 P 扫描 S](2026-10-31-gpu-timing.md) | 240 个正式有效样本，Profiler 独立保存 | `results/timing-*` |
| [2026-11-01：8K–128K Prefix 扫描](2026-11-01-prefix-scan-128k.md) | 480 个缓存有效样本；性能隔离存在监控缺口 | `results/prefix-scan-128k`、`results/prefix128k-*` |
| [2026-11-02：18×9 L1 矩阵](2026-11-02-l1-matrix.md) | 未完成：v1 为 23/1620，v2 启动失败 | `results/l1-matrix-20261102-v1`、`v2` |
| [2026-11-04：本地 dev 镜像小样本 GPU 验收](2026-11-04-dev-smoke.md) | 完成：Cold/L1/L2 各 3 个有效样本，总控验收与清理通过 | `results/dev-smoke-20261104-v3` |

`results/` 被 Git 忽略。克隆代码不会获得原始数据，需另行保留或同步这些目录；记录中的数据链接只在数据存在的工作区可用。
本次整理没有修改旧 CSV、分位数、日志、输入、源码快照或运行配置。

## 源码版本与提交检查点

历史运行先于本次代码整理提交。它们的 SGLang 基点是 `baa9befa381511cd4d56f83980951e5a45fffc6d`；除关闭 HiCache 的初期基线外，还使用了未提交的本地补丁。
准确 Driver 版本由 `config.json` 的 `script_sha256` 和 `source/` 确认；SGLang 版本由 `launch.json` 的 commit/status 及 `sglang.patch` 确认。
不要把新创建的提交写成旧实验当时运行的 commit，也不要改写旧元数据来让它们看起来已经提交。

2026-11-02 创建以下代码检查点：

| 仓库 | commit | 内容 |
|---|---|---|
| `hicache-experiments` | `dc686f1` | 原有 Cold/L1 基线提交，2026-10-30 |
| `sglang` | `02a56e2b2499bbd5c37f17103b4fed2e5121c63f` | 本地缓存 inspect/demote、调度消息、五项回归测试 |
| `hicache-experiments` | `de101ec59aa2b15a050aa3dc367c64e5eebd69dd` | 两脚本的 L2、原生 GPU 计时、Profiler、双路环境监控与矩阵绘图 |

整理前的 SGLang 补丁 SHA256 为 `67f8a5f47e6cacabe55393e905ca54a81f3c40d8d475638d94775efe2fa4ffbc`。
经逐字节比对，`l2-scan`、`timing-scan`、`prefix-scan-128k` 归档的 `sglang.patch` 都与该补丁一致；新 SGLang commit 固定了这份代码。
Driver 经历多个版本，不能用最新提交代替全部历史快照；逐轮记录保留已核对的哈希。

## 历史整理的验证

以下保留旧整理时的环境和验证记录；当前重构验收见上表 2026-11-10 记录。

- 两脚本的模拟 `self-test`、Python/Bash 语法和 Git whitespace 检查通过。
- CPU-only 容器验证通过：旧基线和计时 CSV 的汇总不变，固定 P/固定 S 的坐标与对数延迟轴正确，九面板曲线和四张矩阵热力图布局正确。
- 文档的本地链接和锚点检查通过，结果表及 Profiler 表按原始 CSV 复核；原始 CSV 未修改。
- 七个 SGLang 修改文件通过 Python 语法解析；历史五项 GPU 缓存树测试通过记录见 `l2-scan/validation.json`。
- 主机 UVM 已报告需 OS 重启的全局致命故障，本次整理没有重跑 GPU 测试、重新启动推理服务或恢复矩阵。

后续每轮开始前，提交两个仓库的代码并在当轮记录中写明实际版本。若运行时仍有工作区改动，保留原始 status、补丁和源码快照，明确标记未提交状态。
