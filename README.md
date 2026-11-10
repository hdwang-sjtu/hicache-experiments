# HiCache Prefill 实验

测量已缓存 P tokens 后，再处理 S tokens 的 prefill 时间，支持 cold、GPU L1 命中和 CPU L2 恢复。1K=1024 tokens。

**默认使用 dummy 随机权重、round-robin MoE 路由和 breakable prefill CUDA Graph。** 不下载模型权重，只准备配置、tokenizer 和 processor；仍然执行完整模型计算并占用相应显存。

同一个 P 下，所有 S、预热和重复测量默认共用一次 P 准备。每次 P+S 请求只读取共享 P，新增 KV/GDN 状态在请求结束时释放，因此 S 每次都重新计算。cold/reference 仍完整计算 P+S，集中在准备 P 之前运行。

在已有 SGLang 依赖的环境中，例如 Vast 的 SGLang 容器：

    python3 -m pip install -r requirements-launch.txt
    python3 run_all.py qwen35-35b-matrix \
      --gpu 0,1 --tp 2 --ep 2 --mem-fraction 0.8 \
      --prefix-lengths 16k 24k 32k \
      --suffix-lengths 1024 2048 16384 \
      --groups l1 --repeats 3

先加 --dry-run 可检查最终配置，不下载、不启动、不写目录。只有 Docker 的宿主机增加 --runtime docker，可用 --image lmsysorg/sglang:dev 选择本地镜像。

[仓库结构](docs/architecture.md) · [云实例运行](docs/cloud-task.md) · [分步部署](docs/deployment.md) · [协议与指标](docs/usage.md) · [Qwen3.5](docs/qwen35.md) · [历史记录](docs/experiments/README.md)

## 选择测量模式

| 参数 | 默认 | 其他选择 |
|---|---|---|
| --load-format | dummy：随机初始化，不下载权重 | auto：下载/加载真实权重 |
| --routing | round-robin：确定性的均衡路由 | original：使用模型 router |
| --prefill-graph | breakable：分段 prefill Graph | disabled：eager |
| --prefix-reuse | 开启：同 P 跨 S/重复复用 | --no-prefix-reuse：每次重新准备 P |

例如使用真实权重、原始路由和 eager：

    python3 run_all.py qwen35-35b-matrix \
      --load-format auto --routing original --prefill-graph disabled \
      --model-dir /path/to/Qwen3.5-35B-A3B-FP8 --groups l1

dummy 模型仍保留原来的网络与量化配置。模拟路由会改变专家选择和混合权重，结果应视为该路由策略下的性能，不能直接与真实路由数据混合拟合。

## 入口与配置

| 入口 | 用途 |
|---|---|
| run_all.py | 准备源码/模型文件、启动服务、采样、验收和清理 |
| run_experiment.py run / analyze | 请求已有服务，或离线重新分析 |
| serve.sh | 在当前环境启动服务 |
| launch_server.sh | 在宿主机创建 Docker 容器并启动服务 |

内部实现放在 hicache/；可选遥测和 trace 分析放在 tools/diagnostics.py。一份 configs/experiments/<实验名>.yaml 同时描述模型、服务和测量，不再引用另一份服务 YAML。--list-experiments 或服务入口的 --list-configs 列出相同配置名。

总控分别启动服务和 Driver；Driver 通过 HTTP 请求 SGLang，SGLang 不启动 Driver。native 表示在当前 Python 环境运行，容器内部使用 native 即可。

## Graph 桶自动生成

只需填写 --suffix-lengths，无需重复填写捕获桶。

- L1/L2 的正式 forward 计算 S，所有 P 共用这些桶。
- cold 计算 P+S，额外捕获固定 chunk 分块后的长度。
- 只测 L1 时，准备 P 和 cold reference 可以 eager；正式请求逐 rank 验证 Graph replay。
- 桶一次性捕获，暂不动态追加或自动分批。捕获失败明确报错，不自动切换模式。

Qwen3.5 使用分段 Graph，GDN 线性注意力部分在当前基线中仍走 eager。decode Graph 关闭；输出一个 token 的实验关注 prefill。

## 模型、源码和结果在哪

| 位置 | 内容 |
|---|---|
| --model-dir / MODEL_DIR | 完整模型或配置目录；dummy 只检查/补齐小文件，auto 检查/补齐权重 |
| HF 默认缓存 | 不指定模型目录时使用，通常为 ~/.cache/huggingface/hub；沿用 HF_HOME / HF_HUB_CACHE |
| 项目 .cache/ | 固定 SGLang 源码与补丁缓存，可用 HICACHE_CACHE_DIR 修改 |
| 项目 results/<run-id>/ | 本轮全部结果，可用 RESULTS_DIR 指向持久卷 |

总控使用版本锁中的 SGLang Python 源码并应用实验补丁；镜像提供 Torch、CUDA、kernel 等依赖。不会修改环境里原有的 SGLang checkout。

查看进度：

    tail -f results/<run-id>/driver.log
    cat results/<run-id>/run.json

迁移结果请复制整个运行目录。code/ 是冻结源码，server/ 是启动记录和唯一完整服务日志，samples/ 是 CSV、逐 rank 证据、汇总和图表。输入采用可重建的紧凑归档；成功轮询的完整 /metrics 不写磁盘。

独立启动和远程 HTTP 客户端见 [部署](docs/deployment.md)。云实例销毁、计费回收和结果同步由部署侧管理。

## 验证

    python3 -m unittest discover -s tests
    bash -n serve.sh launch_server.sh

CPU 回归验证协议和编排；GPU 验收与历史性能结论分别记录，不能相互替代。
