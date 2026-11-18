# Airified Strata

[English](README.md) · [简体中文](README.zh-CN.md)

基于 [Strata](https://github.com/Niko1221/Strata) 的本地推理部署，面向 **Qwen3.8-Flash-Next**
混合专家（MoE）模型，包含针对性的引擎优化、调优后的配置、内存专家卸载，以及可跨重启的自动前缀复用。

经过验证的配置在 **两张 22 GiB 的 RTX 2080 Ti** 上运行 **Q4_K_M** 检查点，专家权重从系统内存读取：
完整 256K 上下文、DFlash2 推测解码和图像输入。下文的配置与测量均以此环境为准。

[性能](#性能) · [部署](#部署) · [配置](#配置) · [持久前缀](#持久前缀) · [验证](#验证)

## 目标配置

| 组件 | 默认值 |
| --- | --- |
| 目标模型 | [OrcaRouter Qwen3.8-Flash-Next-Uncensored GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF)，**Q4_K_M**（三个分片） |
| 专家 | 48 层 × 512 个专家，每个 token 路由 10 个；两张卡上各有显存专家缓存，其余由 CPU 从内存计算 |
| KV 缓存 | **INT8**，全部在 GPU 上 |
| 上下文 | **262144 个单元**（引擎内部保留 8 个），包括文本输入、图像输入和输出 |
| 推测解码 | Strata **DFlash2** 草稿包，置信度 0.5 |
| 并行方式 | 双卡**层切分**，固定 24/24 并只加载本卡层的稠密权重；**一次处理一个请求** |
| 预填充 | 自动分块（实测 8192 token） |
| 视觉 | F16 投影器在 **CPU** 上运行；每张图最多 **1024 个视觉 token**（`VISION_MODE=gpu` 时为 4096） |
| 会话复用 | **自动持久前缀快照**，默认开启 |

其他检查点或量化需要各自准备和验证。固定 24/24 切分面向一对对称的 22 GiB 显卡；其他显卡组合请使用[自动切分备选配置](#配置文件)。

## 性能

**简而言之：在同一硬件上，相比按上游 Orca 指南设置运行的上游发布版，解码提升 24–30%，预填充约为 5.4–6.4 倍。与使用相同预填充分块、未修改算子的源码构建相比，预填充提升 35–51%，解码提升 26–30%。**

| 路线 | 32K 代码 | 128K 代码 | 32K 文章 | 128K 文章 |
| --- | ---: | ---: | ---: | ---: |
| 上游发布引擎 0.1.58，Orca 指南设置（512 token 预填充） | 406.0 / 78.75 | 397.5 / 75.30 | 414.6 / 65.62 | 401.7 / 61.61 |
| 未修改算子的源码构建，自动预填充分块 | 1644.1 / 77.0 | 1697.0 / 75.4 | 1648.6 / 64.8 | 1700.2 / 61.8 |
| 本仓库，自动切分备选配置 | 2222 / 89.83 | 2563 / 93.46 | 2236 / 75.10 | 2561 / 73.41 |
| **本仓库，默认配置** | **2220 / 97.55** | **2565 / 97.66** | **2240 / 82.89** | **2565 / 80.09** |

单元格为 **预填充 / 解码 token/s**：**32768 或 131072 个输入 token，1024 个自然输出 token**，贪心采样，正常 EOS，单请求，无前缀复用，不计模型加载和预热。预填充 = 输入 token 数 / 客户端首 token 时间；解码不计第一批流式输出。默认配置为两次全新进程运行的平均值，其他为两次的中位数或单次结果；各行在不同日期和引擎版本上测得，但提示词、显卡组合和计时定义相同。

默认配置在完整窗口文本和图像运行中的峰值：**每张卡约 21.5 GiB**（含驱动保留），**进程常驻内存约 114 GiB**。预填充主要受从内存流式读取专家的限制；解码取决于专家缓存命中率，因此代码与文章结果不同。

## 假定环境

| 项目 | 参考配置 |
| --- | --- |
| GPU | 两张 **22 GiB** RTX 2080 Ti，SM75，参考主机上以 NVLink 互连（其他互连方式未测试） |
| PCIe | 每张显卡需完整的 **PCIe 3.0 x16** 链路；专家权重经由它从内存传输，更窄或更低速的链路会使速度大幅下降 |
| CPU | AMD EPYC 7B13，单路，**NPS4**；其他 x86-64 CPU 可用，但不启用 [NUMA 布局](#自动-epycnps4-cpu-布局) |
| 内存 | 参考主机 **256 GiB** DDR4-3200；PLE 表（32.8 GiB）驻留内存时，服务峰值约 **114 GiB** |
| 存储 | 分片与投影器约 **120 GB**，pack 约 1.5 GB，另加持久前缀预算（默认 64 GiB） |
| 系统 | Linux x86-64；Ubuntu 26.04.1 LTS，NVIDIA 驱动 615.71.09 |
| CUDA / 编译器 | CUDA **13.4**，GCC **15.2** |
| 构建工具 | CMake **4.2.3**、Ninja、Python **3.10+**（已测试 3.13.2） |
| Strata 基线 | [`13eedf55`](https://github.com/Niko1221/Strata/tree/13eedf552417a6a791a7d13c6597e24300ee3a09) 加 [36 个本地补丁](upstream.json)；llama.cpp/ggml [`3cf03257`](https://github.com/ggml-org/llama.cpp/tree/3cf03257f219afbe7334045ff7c6a06ac68c627d) |

**普通 11 GiB 的 2080 Ti 未经验证。** 构建目标为 `75-real`；其他 GPU 架构和工具链不在这些结果范围内。低于约 128 GiB 的内存未测试。

## 功能与优化

| 改动 | 收益 |
| --- | --- |
| [原生超连接读取](patches/hc-native-blocks.patch)（[不保留 BF16 副本](patches/hc-native-no-bf16.patch)） | 解码直接读取检查点自身的 Q4_K/Q6_K/Q5_0 混合权重（约为 BF16 字节量的三分之一）；释放的显存交给专家缓存。 |
| [SM75 张量核心投影](patches/bf16-tc-operator-scope.patch)与 [FP16 图像输入](patches/bf16-tc-direct-fp16-input.patch) | Turing 上以 FP32 累加的 BF16→FP16 张量核心矩阵乘。 |
| [分组 K-quant 专家](patches/kquant-grouped-gu-two-rows.patch)、[Q5_0 down 覆盖](patches/q5_0-grouped-expert-coverage.patch) | Q4_K_M 的专家格式运行在分组原生内核上。 |
| [SM75 启动融合](patches/sm75-cuda-small-fusions.patch)、[输出头预取](patches/sm75-head-prefetch-dispatch.patch) | 每层更少的小内核；更快的输出头读取。 |
| [多阶段会话](patches/multi-stage-native-session-runtime.patch)、[层切分下的 DFlash2](patches/layer-split-dflash-prompt-taps.patch) | 原生会话保存/恢复和草稿模型可跨两个 GPU 阶段工作。 |
| [自动持久前缀](patches/automatic-persistent-prefix-cache.patch) | 跨重启复用共享的提示前缀，包含检查点、淘汰和安全收尾。 |
| [EPYC/NPS4 专家布局](patches/nps4-expert-row-placement.patch)、[服务生命周期](patches/epyc-nps4-service-lifecycle.patch) | 专家行与 CPU 工作线程留在同一 NUMA 节点。 |
| [有界图像键](patches/bounded-image-wire-keys.patch) | 在原生解析前拒绝格式错误的图像标识。 |

全部补丁及其启用方式见 [patches/README.md](patches/README.md)。其中若干是显式改变舍入的算子选择，由配置文件启用；整体请求结果与上游并非逐位一致。

## 部署

### 1. 安装依赖并克隆

安装 CUDA 13.4 和兼容驱动（[NVIDIA 指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)），然后：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build python3 python3-venv pkg-config

git clone --recurse-submodules https://github.com/airium/strata-airified.git
cd strata-airified
```

子模块已固定版本：**不要更新到上游 `main`**。`build.sh` 会应用补丁，并拒绝意外的版本或冲突。

### 2. 创建 Python 环境

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r vendor/strata/requirements.txt 'huggingface_hub[cli]'
```

### 3. 下载模型、投影器和草稿包

只把以下文件下载到同级的 `models/` 目录（约 120 GB）。如有需要，先运行 `.venv/bin/hf auth login`。

```bash
MODEL_ROOT="$(pwd)/../models"
.venv/bin/hf download orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF \
  Qwen3.8-Flash-Next-Uncensored-Q4_K_M-00001-of-00003.gguf \
  Qwen3.8-Flash-Next-Uncensored-Q4_K_M-00002-of-00003.gguf \
  Qwen3.8-Flash-Next-Uncensored-Q4_K_M-00003-of-00003.gguf \
  mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf \
  --local-dir "$MODEL_ROOT/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF"

mkdir -p runtime/dflash2
curl -L -o runtime/dflash2/qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack \
  https://github.com/Niko1221/Strata/releases/download/v0.2.12/qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack
echo '4df2e5ab2b47c064afbe662e16e4f06569877125c2d0e2123a4ac755266c664c  runtime/dflash2/qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack' | sha256sum -c
```

草稿包按发布原样使用。三个分片需放在一起；投影器保持 F16。

### 4. 构建

```bash
bash build.sh          # 引擎（增量构建；--rebuild 为全新构建）
bash build-vision.sh   # 图像读取器
```

默认值：`CUDA_TOOLKIT=/usr/local/cuda-13.4`、`BUILD_DIR=build/sm75`、`BUILD_JOBS=16`（读取器为 8）。目录移动后会自动刷新过期的 CMake 路径；移动仓库后请重建 `.venv`。`build/sm75/BUILD.json` 记录源码、补丁和二进制哈希。

### 5. 生成 pack

Strata 以 BF16 读取少量小投影，其余直接从 GGUF 读取。只需生成一次兼容 pack（约 1.5 GB；不复制专家权重）：

```bash
STRATA_GGUF_PY="$PWD/vendor/llama.cpp/gguf-py" .venv/bin/python vendor/strata/tools/iq_pack.py \
  --gguf "$MODEL_ROOT/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/Qwen3.8-Flash-Next-Uncensored-Q4_K_M-00001-of-00003.gguf" \
  --out runtime/packs/orca-q4_k_m --compat-bf16
```

该命令同时把模型的分词器导出到 pack 中。背景说明见上游 [Orca 流程](vendor/strata/docs/ORCA.md)。

### 6. 选择显卡并启动

```bash
test -f config.env || cp config.example.env config.env
```

**先编辑 `config.env`。** 把 `GPU_DEVICES` 设为你的显卡组合，双卡主机通常为 `0,1`；模板使用 `2,3`。模型、pack 和草稿包路径与上面的步骤一致。

```bash
bash serve.sh --print-config
bash serve.sh                 # 图像读取器按 config.env 中的 VISION_MODE（模板为 cpu）
bash serve.sh --vision gpu    # 仅本次启动生效：off、cpu 或 gpu
```

服务在前台运行于 `http://127.0.0.1:3893/v1`；按 Ctrl-C 停止。启动时会把约 75 GiB 专家载入内存并填充显存专家缓存，需要几分钟。**请以普通用户运行，不要使用 `sudo`**（见[故障排查](#故障排查)）。

## 配置

主机设置写在被忽略的 `config.env` 中（[模板](config.example.env)）；显式设置优先于内置默认值和配置文件。

| 设置 | 默认值 / 用途 |
| --- | --- |
| `PROFILE` | `q4km-int8kv-256k-dflash2-sm75-tcwide-c50-trim24-fused-hcnative-persistent` |
| `GPU_DEVICES` | 模板为 `2,3`；**请改为你的显卡组合** |
| `HOST` / `PORT` | `127.0.0.1` / `3893` |
| `SERVED_NAME` | `orcarouter-qwen3.8-flash-next-q4-k-m` |
| `MODEL_ROOT` | 同级 `../models`；`TARGET_MODEL`（第一个分片）和 `MMPROJ` 可单独覆盖 |
| `PACK_DIR` / `DRAFT_PACK` | `runtime/packs/orca-q4_k_m` / `runtime/dflash2/…selq4.pack` |
| `PYTHON_BIN` | 模板为 `.venv/bin/python` |
| `CPU_NUMA_POLICY` | `auto`；`off` 关闭 EPYC/NPS4 布局 |
| `PERSISTENT_CACHE_DIR` / `_GIB` / `_MIN_TOKENS` | 同级 `../strata-kv-caches` / `64` / `1024` |
| `VISION_MODE` / `VISION_MAX_TOKENS` | 模板为 `cpu`（可选 `off`、`cpu`、`gpu`）/ `auto`（CPU 1024，GPU 4096） |
| `RUNTIME_DIR` | `runtime/`：解析后的配置、日志和租约记录 |

### 配置文件

| 配置 | 用途 |
| --- | --- |
| `…-trim24-fused-hcnative-persistent`（**默认**） | 固定 24/24 切分、按阶段加载稠密权重、原生超连接读取、SM75 融合、持久前缀。适用于对称的 22 GiB 双卡。 |
| `…-trim24-fused-hcnative-kqmmq-direct-persistent`（可选） | 默认路线基础上，Q4_K 专家的提示词乘法改用 int8 MMQ（`STRATA_MMQ_KQ=1`），并原地读取已缓存的专家（`STRATA_MMQ_DIRECT=1`）。预填充约快 12%（32K/128K，散文/代码），解码不变，但参考检查中质量有可测下降（下一个词元准确率 100 个中 73 → 71，gold NLL +0.014）：适合能接受这一代价、以长提示词为主的工作。 |
| `…-sm75-tcwide-c50-persistent`（**备选**） | 自动层切分，使用相同的 SM75 算子和持久前缀。适用于其他显卡组合。 |
| `…-trim24-fused-persistent` | 默认路线，但不使用原生超连接读取。 |
| `…-sm75-tcwide-c50` | 备选路线，但不使用持久前缀。 |
| `q4km-int8kv-256k-dflash2` | 原始算子基线，DFlash 置信度 0.35。 |

所有配置共用检查点、INT8 KV、262144 个单元和 DFlash2。SM75 算子会改变舍入，不同配置的显存驻留也不同，因此彼此输出并非逐位一致。更换检查点、工具链或硬件后，请重新验证质量、内存和速度。

## 持久前缀

`-persistent` 配置会在请求完成时保存原生快照，并在后续请求（包括重启后）复用最长的兼容前缀，只需预填充未缓存的后缀。快照包含 INT8 KV、循环/索引器状态、各 GPU 检查点和 DFlash2 谱系；每 16K token 一个检查点，因此修改问题后可从较早的检查点继续。

- `PERSISTENT_CACHE_DIR` 以私有权限创建并包含对话内容；请放在仓库之外。`prefixes/` 为自动使用；`sessions/` 用于显式的槽位保存/恢复。
- 已提交的快照共享类似 LRU 的预算（`PERSISTENT_CACHE_GIB`）；替换时需要临时额外空间。保存和恢复会在请求边界增加延迟。
- 构建、算子、模型、KV、上下文和层布局的变化会选择不同的命名空间；损坏或不兼容的文件会回退为正常读取。
- `"cache_prompt": false` 跳过复用和保存；`"strata_checkpoint": false` 只复用不保存。带图像的请求跳过磁盘层（重复的图像仍会复用进程内缓存）。

复用以响应中的原生计数（`prompt_tokens_details.cached_tokens`）为准，而不是文件是否被创建。

## 图像输入

`VISION_MODE=cpu` 在 CPU 上运行 F16 投影器（`VISION_THREADS`，默认 16），不占用显存。`VISION_MODE=gpu` 改为在 `VISION_CUDA_DEVICE`（默认第一张文本显卡）上编码。读取器先于引擎启动，因此该卡的专家缓存会相应变小：在默认配置和 4096 token 预算下，第一张卡保留 5521 个槽位中的 4988 个，预填充不变，解码约降低 1–3.5%，而 4000 个 token 的图像编码约 6 秒，CPU 上则超过 1 分钟。以文本为主时选 `cpu`，图像多或图像大时选 `gpu`；切换只需重启，`bash serve.sh --vision gpu`（或 `cpu`、`off`）可在单次启动中覆盖 `config.env`。

`VISION_MAX_TOKENS` 是单张图像的分辨率预算，而非固定数量。`auto`（默认）在 CPU 上为 1024、在 GPU 上为 4096；填写数字则覆盖。编码时间的增长快于 token 数（16 个 CPU 线程上 1000 个 token 约 7 秒、4000 个超过 1 分钟；单张 2080 Ti 上 4000 个不到 6 秒）。较小的图像保持原尺寸。图像 token 计入 262144 个单元的上下文。`/health` 会报告 `"images": true`；手动定义模型能力的客户端需要启用图像输入。

## 自动 EPYC/NPS4 CPU 布局

`CPU_NUMA_POLICY=auto` 时，启动器检查服务 cgroup 限制内是否为单路 AMD EPYC、四个均衡的 CPU 与内存 NUMA 节点，以及支持 NPS4 的构建。只有满足时才按节点分配专家行和 CPU 工作线程。Intel、Ryzen、双路、其他节点数和受限的 cgroup 保持默认路线；`--print-config` 在 `cpu_numa_policy` 下显示判断结果。

如果有 `numad` 守护进程在运行，只有引擎进程会在加载模型前被排除，并由一个看护进程在引擎退出时撤销排除。这会通过 `sudo -n` 调用 numad 的客户端，因此该用户需要对它有免密 sudo 权限；否则请设置 `CPU_NUMA_POLICY=off`。没有 numad 时无需任何权限。修改在下次启动时生效。

## API

兼容 OpenAI 的端点：

```bash
curl http://127.0.0.1:3893/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"orcarouter-qwen3.8-flash-next-q4-k-m","messages":[{"role":"user","content":"Explain binary search briefly."}],"max_tokens":256,"temperature":0}'
```

图像使用 `image_url` 内容（data URL）。`"reasoning_effort":"none"` 可为单个请求关闭思考。端点：`/health`、`/v1/models`、`/v1/chat/completions`、`/slots`、`/metrics`。

## 验证

在参考主机上以默认配置通过：

- **完整窗口：** 261112 个输入 + 1024 个自然输出 token，填满 262136 个可用单元。
- **完整窗口图像：** 261112 token 的图像提示读取正确；重复图像复用 261105 个 token；更换图像被正确识别（复用 245760，读出新标签）。
- **重启复用：** 全新进程复用 4926（文章）和 5641（代码）个前缀 token，原生计数与 API 一致；关闭复用、手动保存/恢复和图像输入也通过。
- **质量：** 31 项功能任务和 8 个执行验证的生成函数；25 篇文档共 12295 个 token 的教师强制下一 token 损失与上一路线相差在 +0.01 nats 以内（+0.0008，95% 区间 −0.0028 至 +0.0043）。
- 封装测试：`.venv/bin/python -m unittest discover -s tests`。

这些是单台主机上的功能、容量和有限的质量检查，不是全面的模型评测，也不保证适用于其他硬件或检查点。

## 故障排查

- **不要使用 `sudo`。** 服务的任何部分都不需要 root。以 root 运行会产生属于 root 的缓存和租约文件，之后普通用户运行时无法复用或清理，并且模型服务会以 root 身份运行。
- **启动时显存不足：** 有其他进程占用显卡，或显卡小于 22 GiB；请使用备选配置并检查 `--print-config`。
- **内存不足：** 仅专家就约 75 GiB；请关闭其他大型进程。
- **`numad exclusion requires noninteractive permission`：** 设置 `CPU_NUMA_POLICY=off`，或允许免密 `sudo` 调用 numad 客户端。
- 启动日志会报告专家缓存大小、层范围，以及默认配置下的 `STRATA_HC_NATIVE=1: … read natively`。

## 开发

补丁流程、实验工作树、测试和诊断：[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。

## 上游与模型署名

封装代码和补丁：[MIT](LICENSE)。[Strata](https://github.com/Niko1221/Strata)（MIT）和 [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT）在子模块中保留各自的许可证；[upstream.json](upstream.json) 记录固定版本和补丁顺序。权重请从 OrcaRouter 单独下载，草稿包来自 Strata 发布页；许可以各自页面为准。结果仅适用于所列检查点和设置，不代表所有 Qwen 变体。
