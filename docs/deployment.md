# 部署与独立入口

[工程入口](../README.md) · [一键云任务](cloud-task.md) · [架构](architecture.md)

完整任务优先使用 run_all.py；下列入口适用于单独开服务、复用现有服务或远程采样。文件职责见架构文档。

## 命名服务配置

所有入口读取 configs/experiments/<名称>.yaml。模型、服务参数、P/S 在同一份配置，不再使用 configs/serve。服务入口的 --list-configs 与总控 --list-experiments 列出相同名称。

服务设置位于 server.sglang，服务环境位于 server.env。server 的 load-format、routing、prefill-graph 默认分别为 dummy、round-robin、breakable。test 中的正式网格决定启动时的 Graph 桶。

服务原生 CLI > 显式部署环境 > YAML；总控先完成 CLI 覆盖和容量检查，再传冻结配置给启动器，避免二次覆盖。不要叠加第二个 --config。

## 准备源码与模型文件

独立服务入口不下载文件。先设置：

    export MODEL_DIR=/path/to/prepared/model-or-assets
    export SGLANG_DIR=/path/to/patched/sglang

MODEL_DIR 在 dummy 下可只含 config、tokenizer、processor；真实模式须有完整权重。不再支持 MODEL_ROOT。显式 SGLANG_DIR='' 使用已安装包；未设置时仍尝试相邻 ../sglang。所用代码必须包含项目所需的实验补丁。

可通过准备工具单独获取文件，revision 使用实验 YAML 中的固定值：

    python3 -m hicache.prepare model \
      --repo Qwen/Qwen3.5-35B-A3B-FP8 \
      --revision 9d1823d2dee688a6b25e77009dc727688c44936e \
      --load-format dummy --directory /path/to/assets \
      --output /path/to/model-info.json

源码准备、依赖与缓存规则见 [云任务](cloud-task.md#固定版本与缓存)。

## 启动服务

已有 SGLang Python 依赖：

    RUN_ID=server-demo GPU=0,1 \
      bash serve.sh qwen35-35b-matrix --tp-size 2 --ep-size 2

宿主机 Docker：

    RUN_ID=server-demo GPU=0,1 IMAGE=lmsysorg/sglang:dev \
      bash launch_server.sh qwen35-35b-matrix --tp-size 2 --ep-size 2

可先加 --dry-run，检查配置和命令而不创建结果。独立服务使用 YAML 的测试长度生成桶；手工改测试网格时应同步修改配置或显式覆盖原生 Graph 桶/容量参数。

RESULTS_DIR/RUN_ID 默认项目 results/UTC时间。启动记录包含实际命令、最终配置、源码、镜像/GPU 证据和 server.log。原始合并 YAML 保存为 profile.yaml。

Docker 模型、代码、源码只读挂载，结果可写；HF 快照挂载所属缓存根。退出只清理本脚本创建的容器，不操作其他容器。

## 远程客户端

客户端只需匹配的 tokenizer 文件与客户端依赖，不需要权重、GPU 或 SGLang。复制 Python 入口及 hicache/、tools/ 包，或直接 clone 本仓库：

    python3 -m pip install -r requirements-client.txt
    python3 run_experiment.py run \
      --base-url http://127.0.0.1:39081 \
      --tokenizer /client/path/to/assets \
      --output-dir results/client-demo \
      --prefix-lengths 2k --suffix-lengths 128 512 \
      --groups cold l1 l2 --repeats 3 --gpu-timing \
      --prefill-graph breakable

服务应独占、单请求。Driver 不重配运行中的服务；填写的 P/S 必须受现有容量和 Graph 桶支持。--server-artifacts 指向客户端可读的 launch.json 所在目录或副本，用于关联来源。

实验缓存控制接口仅接受服务端 loopback。远程使用 SSH 隧道：

    ssh -N -L 39081:127.0.0.1:39081 user@gpu-host

客户端的 tokenizer/output-dir 是客户端路径；--server-profile-dir 是服务端路径，不要求共享文件系统。不能把跨主机单调时钟直接用于遥测关联。
