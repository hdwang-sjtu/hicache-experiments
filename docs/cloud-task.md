# 云实例一键实验

[入口](../README.md) · [架构](architecture.md) · [Qwen3.5](qwen35.md)

在提供 GPU、足够磁盘与内存的实例上启动 SGLang 镜像，进入终端后：

    git clone https://github.com/hdwang-sjtu/hicache-experiments.git
    cd hicache-experiments
    python3 -m pip install -r requirements-launch.txt
    python3 run_all.py qwen35-35b-matrix \
      --runtime native --tp 2 --ep 2 --mem-fraction 0.8 \
      --prefix-lengths 16k 24k 32k \
      --suffix-lengths 1024 2048 16384 \
      --groups l1 --repeats 3

容器内部用 native，继承平台已经提供的 GPU 可见范围。宿主机需要项目创建容器时用 --runtime docker，并通过 --gpu 0,1 等显式选择设备。

默认 dummy+round-robin+breakable Graph。第一次只下载模型小文件，不下载权重；完整模型仍在 GPU 上分配并计算。运行真实权重增加 --load-format auto，使用模型原路由增加 --routing original。

## 启动后的过程

1. 解析同一份实验 YAML，合并 CLI，检查 P/S、chunk、容量和拓扑，生成正式请求的 Graph 桶。
2. 创建独立结果目录，冻结完整工具包与配置，检查端口。
3. 复用或获取版本锁中的 SGLang 源码并应用实验补丁；准备 HF 小文件或真实权重。
4. 启动服务、等待健康、核对本轮服务身份。
5. Driver 逐点做 reference、预热和正式采样，检查全部 rank 的缓存、计算量和执行路径。
6. 重算结果并核对各组有效样本数，停止本轮服务或删除本轮容器。

脚本不自动安装 Torch/CUDA/kernel 依赖，不停止云实例计费。官方镜像提供依赖，项目固定源码提供实验接口；不能仅凭镜像标签判断实际运行代码版本。

## 固定版本与缓存

模型仓库/revision 在 configs/experiments/ 中，SGLang 基线/补丁/镜像在 configs/runtime.lock.json 中。

- 不指定模型路径：HF 库选择默认缓存，通常为 ~/.cache/huggingface/hub；HF_HOME / HF_HUB_CACHE 按 HF 原生规则生效。
- --model-dir 或 MODEL_DIR：使用此完整目录。dummy 只检查小文件；auto 检查 safetensors 索引、分片和头部并补齐缺失文件，不做全权重内容哈希。
- HICACHE_CACHE_DIR：默认项目 .cache，存放源码 checkout 与准备锁，不是 HF 模型目录。
- RESULTS_DIR：默认项目 results，可以指向持久卷。

源码缓存复用前检查固定 HEAD 与补丁差异；不会静默覆盖不匹配缓存。HF 快照可能通过两层链接引用共享 blobs，Docker 挂载缓存根保留链接结构。HF_TOKEN 等下载认证仍由 HF 库处理。

## 结果、失败与清理

指定 --run-id 为本轮起名；已有目录不会覆盖。默认名称由 UTC 时间和实验名组成。

    results/<run-id>/
      run.json                   总控状态与失败阶段
      resolved-experiment.json   CLI 覆盖后的实验配置
      resolved-serve.json         实际下发的服务配置
      model.json / source.json   文件和源码来源
      code/                      冻结工具包
      server/                    启动记录、GPU 信息和 server.log
      samples/                   输入清单、响应、CSV、逐 rank 证据、图表
      prepare.log / driver.log   准备与采样进度
      gpu-memory.jsonl           采样得到的显存记录

server-launch.log 不重复保存完整服务输出，排障看 server/server.log。Graph 启动失败不会自动降级；正式请求回退 eager 会保留证据并失败，不混入有效样本。

    tail -f results/<run-id>/driver.log
    cat results/<run-id>/run.json

退出或中断会清理本轮拥有的进程组/容器；不操作已有服务或其他容器。服务清理不等于删除云实例。正式样本暂不支持自动续跑，重新运行请使用新 run-id。

迁移或 rsync 同步时复制整个运行目录。仓库不包含原始结果、模型或缓存。
