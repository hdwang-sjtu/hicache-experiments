# 2026-11-05：统一入口的 Qwen3.5 Cold/L1/L2 与 TP/EP 验收

[使用方法](../qwen35.md) · [实验索引](README.md)

## 已完成的 GPU 矩阵

使用本地 `lmsysorg/sglang:dev`、四张 RTX 6000 Ada（每卡 49140 MiB），本地只读模型目录。通过容器内 `run_all.py --runtime native` 启动，所有任务结果写到宿主挂载目录，然后复制到本项目 `results/unified-qwen35/`。

| 模型 | TP/EP | P / S | 有效样本 | 原始证据 |
|---|---|---|---:|---|
| 35B-A3B-FP8 | 1/1 | P=2K、4K；S=128、512 | 36 | [35-tp1ep1-v2](../../results/unified-qwen35/35-tp1ep1-v2/) |
| 35B-A3B-FP8 | 2/1 | 同上 | 36 | [35-tp2ep1-v1](../../results/unified-qwen35/35-tp2ep1-v1/) |
| 35B-A3B-FP8 | 2/2 | 同上 | 36 | [35-tp2ep2-v1](../../results/unified-qwen35/35-tp2ep2-v1/) |
| 35B-A3B-FP8 | 4/1 | 同上 | 36 | [35-tp4ep1-v1](../../results/unified-qwen35/35-tp4ep1-v1/) |
| 35B-A3B-FP8 | 4/4 | 同上 | 36 | [35-tp4ep4-v1](../../results/unified-qwen35/35-tp4ep4-v1/) |
| 122B-A10B-FP8 | 4/4 | P=128K，S=2048，chunk=32K | 9 | [122-tp4ep4-v1](../../results/unified-qwen35/122-tp4ep4-v1/) |

每个 P/S/组采集 3 个有效样本，warmup 和准备请求不计入上表。所有任务均完成结果验收与所属服务清理；每个请求的所有 rank 指标完整，Cold/L1/L2 输出 token 一致。35B 共 180 个、122B 共 9 个正式样本。

这轮运行的代码快照先于最终主项目提交，不能把最终提交当作每轮运行时的 HEAD。逐轮 `code/`、`run.json` 中的代码哈希、`sglang.patch`、原始配置和状态均保持原样。SGLang 运行补丁与后续提交 `b511bd1902` 的对应基线差异一致。

宿主机 `run_all.py --runtime docker --image lmsysorg/sglang:dev` 也单独通过 35B TP2/EP2、P=2K/S=128 的三组各 3 次验收，见 [35-host-docker-tp2ep2-v1](../../results/unified-qwen35/35-host-docker-tp2ep2-v1/)。该轮核对了实际 image ID、双卡设备选择、只读模型/源码挂载、可写结果挂载、UID/GID 和容器清理；结果与原始指标的独立复核通过。

## 122B 长前缀证据

TP4/EP4、mem_fraction=0.8、BF16 KV、context/max-total-tokens=135168、chunk=32768、每 rank HiCache 预算 8 GB、max-mamba-cache-size=32。实际报告 KV 容量为 135168，足够 P+S+1=133121。

独立审计重新解析 `responses.jsonl` 中的原始 Prometheus 与 HTTP 响应，并与逐 rank 记录、CSV 和代码快照对应：

- 准备 P 为 4 次 forward；Cold 为 5 次；L1/L2 后缀均为 1 次，每个 rank 都满足。
- Cold 无缓存命中；L1 device hit=P；L2 host hit=P，且执行前 GPU KV 和 P 的线性状态均已驱逐，主机备份仍完整。
- L2 恢复字节逐 rank 等于 P×该池每 token 字节数，加实际线性状态副本数×状态大小；Cold/L1 无缓存 H2D。
- 同一输入输出一致，无回撤，计时取所有 rank 的最大值。

| 组别 | HTTP 中位数（ms） | max-rank Forward 中位数（ms） | max-rank H2D 中位数（ms） |
|---|---:|---:|---:|
| Cold | 22572.44 | 21773.39 | 0 |
| L1 | 495.15 | 403.28 | 0 |
| L2 | 522.64 | 421.82 | 117.41 |

四卡最高采样显存为 39339 / 39337 / 39291 / 39291 MiB。Forward 是 CUDA 流区间，包含通信和等待；H2D 可以与计算重叠，不能把两个时间直接相加。每组仅 3 次，用于功能验收，不据此推断生产环境尾延迟或对比旧版本性能。

[独立审计汇总与脚本](../../results/unified-qwen35/verification/)保留原始指标复核方法；每轮另有 `independent-audit.json`。

## 回归与清理

- 19 项 CPU 回归及 Driver self-test 通过：包括全 rank 预检协议、失败路径、延迟/缺失指标、模型只读检查、容量边界和仅选 L1 时的 Cold 对照。
- 8 项真实 GPU 缓存树测试通过，包括原有 FULL-KV 与 FULL+MAMBA 的槽位释放、备份保留、锁检查和重复驱逐。
- 原有 [Qwen3-8B 回归](../../results/unified-qwen35/qwen3-8b-regression-v1/)也通过 Cold/L1/L2 各 3 个有效样本，原始指标与 CSV 独立复核一致。
- Python/Bash 语法、Markdown 本地链接/锚点/代码块、Git whitespace 和补丁与源码提交的一致性检查通过。既有 Mermaid 调用图未改变，仍由总控分别启动服务和 Driver。
- 本轮共 207 个正式有效样本，另保留一轮无有效样本的失败记录。归档逐文件与挂载目录核对一致，原始记录未改写。
- 最终四卡均恢复到 11 MiB、利用率 0%，本轮容器均已清理，其他用户容器保留。验证概要及清理检查见 [verification](../../results/unified-qwen35/verification/)。

## 版本与运行边界

- 镜像 ID：`sha256:a1dc7080e3aa146f5d6ef1744f973b0591d8760d694f19badd9adce6f3fc6303`。
- 镜像摘要：`lmsysorg/sglang@sha256:b4323a896f349d43a267c419c0267538b06c64a73110a0d0fd87496234df5dcd`。
- 实际 SGLang Python 来源：基线 `baa9befa381511cd4d56f83980951e5a45fffc6d` 加本仓库补丁；不是镜像中安装包的源码。
- 成功运行补丁 SHA256：`45913465ca6b28a7c01c2a3dd1a186adcc52367728c9d2d691af3fc712a20d0f`。
- Torch 2.14.1+cu130、CUDA 13.0、Transformers 5.19.0；具体包与服务配置见逐轮启动记录。
- 本地模型 revision 线索：35B 为 `9d1823d2dee688a6b25e77009dc727688c44936e`，122B 为 `a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9`，来自本地 HF metadata，不代表全权重内容哈希验证。

源码从本机 Git 仓库离线准备到 `/tmp/hicache-unified-20261105/data/cache`，基线和补丁仍按版本锁验证。本机 NFS 私有目录不能直接交给 Docker daemon 挂载，因此本轮从 `/tmp` 暂存启动，结果回存工作区。这个限制与模型或缓存正确性无关。

本次未租用 Vast 实例；两张/四张 PRO6000 的平台验收仍需在目标机器执行。CPU 回归、缓存树 GPU 测试、真实模型实验和目标平台验收分别提供不同证据。

## 失败与修复

[35-tp1ep1-v1](../../results/unified-qwen35/35-tp1ep1-v1/)在首个缓存控制请求失败：新 Scheduler 接入误用了不存在的 `self.tp_rank`。修正为当前并行配置的 `tp_rank` 后重新执行 v2。失败轮没有正式有效样本，总控保存失败阶段、日志和结果并清理服务；没有把它计入成功矩阵。

`hicache-experiments-vast-l1` 的工作区、分支、代码和结果未删除或搬迁。其分块、容量和指标等待经验通过主项目实现及[固定历史记录链接](../qwen35.md#从临时-l1-项目吸收的经验)保留，不作为当前运行时依赖。
