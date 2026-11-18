# Airified Strata

[English](README.md) · [简体中文](README.zh-CN.md)

A [Strata](https://github.com/Niko1221/Strata)-based local inference deployment for the
**Qwen3.8-Flash-Next** mixture-of-experts model, with targeted engine optimizations, tuned profiles,
RAM expert offload and automatic restart-surviving prefix reuse.

The validated profile runs the **Q4_K_M** checkpoint on **two RTX 2080 Ti 22-GiB cards** with the
experts streamed from system RAM: full 256K context, DFlash2 speculative decoding and image input.
Configuration and measurements below refer to this setup.

[Performance](#performance) · [Setup](#setup) · [Configuration](#configuration) · [Persistent prefixes](#persistent-prefixes) · [Validation](#validation)

## Target configuration

| Component | Default |
| --- | --- |
| Target model | [OrcaRouter Qwen3.8-Flash-Next-Uncensored GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF), **Q4_K_M** (three shards) |
| Experts | 48 layers × 512 experts, 10 routed per token; a VRAM expert cache on both GPUs, the rest computed from RAM by the CPU |
| KV cache | **INT8**, fully on GPU |
| Context | **262144 cells** (eight reserved by the engine), incl. text input, image input and output |
| Speculative decoding | Strata **DFlash2** drafter pack, confidence 0.5 |
| Parallelism | Two-GPU **layer split**, fixed 24/24 with stage-local dense weights; **one request at a time** |
| Prefill | Automatic chunking (8192 tokens observed) |
| Vision | F16 projector on the **CPU**; up to **1024 visual tokens per image** (4096 with `VISION_MODE=gpu`) |
| Session reuse | **Automatic persistent prefix snapshots**, enabled by default |

Other checkpoints or quants need their own preparation and qualification. The fixed 24/24 split is
for a symmetric pair of 22 GiB cards; use the [automatic-split fallback](#profiles) on other pairs.

## Performance

**TL;DR: decode +24–30% and prefill about 5.4–6.4× over the upstream release run with its Orca guide's
settings on the same hardware. Against an unmodified source build with the same prefill chunking, prefill is
+35–51% and decode +26–30%.**

| Route | 32K code | 128K code | 32K prose | 128K prose |
| --- | ---: | ---: | ---: | ---: |
| Upstream release engine 0.1.58, Orca guide settings (512-token prefill) | 406.0 / 78.75 | 397.5 / 75.30 | 414.6 / 65.62 | 401.7 / 61.61 |
| Unmodified-operator source build, automatic prefill chunking | 1644.1 / 77.0 | 1697.0 / 75.4 | 1648.6 / 64.8 | 1700.2 / 61.8 |
| This repo, automatic-split fallback profile | 2222 / 89.83 | 2563 / 93.46 | 2236 / 75.10 | 2561 / 73.41 |
| **This repo, default profile** | **2220 / 97.55** | **2565 / 97.66** | **2240 / 82.89** | **2565 / 80.09** |

Cells show **prefill / decode tokens/s**: **32768 or 131072 input tokens, 1024 natural output
tokens**, greedy sampling, normal EOS, one request, no prefix reuse, model loading and a warm-up
excluded. Prefill = input tokens / client time to first token; decode excludes the first streamed
batch. Default-profile cells are the mean of two fresh-process runs, others medians of two or a
single pass; rows were measured on different dates and engine pins, with the same prompts, GPU pair
and timing definitions.

Peak usage of the default profile during full-window text and image runs: **about 21.5 GiB per GPU**
(including driver reservation) and **about 114 GiB process RSS**. Prefill is dominated by streaming
the experts from RAM; decode depends on the expert-cache hit rate, so code and prose differ.

## Assumed environment

| Item | Reference configuration |
| --- | --- |
| GPUs | Two **22 GiB** RTX 2080 Ti cards, SM75, NVLink-connected on the reference host (other interconnects untested) |
| PCIe | A full **PCIe 3.0 x16** link per GPU; experts stream from RAM over it, so narrower or slower links cut speed sharply |
| CPU | AMD EPYC 7B13, one socket, **NPS4**; other x86-64 CPUs work without the [NUMA placement](#automatic-epycnps4-cpu-placement) |
| RAM | **256 GiB** DDR4-3200 on the reference host; observed peak **about 114 GiB** for the server with the 32.8 GiB PLE table held in RAM |
| Storage | About **120 GB** for the shards and projector, 1.5 GB for the pack, plus the persistent-prefix budget (64 GiB by default) |
| Host | Linux x86-64; Ubuntu 26.04.1 LTS, NVIDIA driver 615.71.09 |
| CUDA / compiler | CUDA **13.4**, GCC **15.2** |
| Build tools | CMake **4.2.3**, Ninja, Python **3.10+** (3.13.2 tested) |
| Strata base | [`13eedf55`](https://github.com/Niko1221/Strata/tree/13eedf552417a6a791a7d13c6597e24300ee3a09) plus [36 local patches](upstream.json); llama.cpp/ggml [`3cf03257`](https://github.com/ggml-org/llama.cpp/tree/3cf03257f219afbe7334045ff7c6a06ac68c627d) |

**Ordinary 11-GiB 2080 Ti cards are not qualified.** The build targets `75-real`; other GPU
architectures and toolchains are not covered by these results. RAM below about 128 GiB is untested.

## Features and optimizations

| Change | Benefit |
| --- | --- |
| [Native hyper-connection reads](patches/hc-native-blocks.patch) ([no BF16 copies](patches/hc-native-no-bf16.patch)) | Decode reads the checkpoint's own Q4_K/Q6_K/Q5_0 mixing weights (about one third of the BF16 bytes); the freed VRAM goes to the expert cache. |
| [SM75 tensor-core projections](patches/bf16-tc-operator-scope.patch) and [FP16 image input](patches/bf16-tc-direct-fp16-input.patch) | BF16→FP16 tensor-core matrix products with FP32 accumulation on Turing. |
| [Grouped K-quant experts](patches/kquant-grouped-gu-two-rows.patch), [Q5_0 down coverage](patches/q5_0-grouped-expert-coverage.patch) | The Q4_K_M expert formats run on the grouped native kernels. |
| [SM75 launch fusions](patches/sm75-cuda-small-fusions.patch), [head prefetch](patches/sm75-head-prefetch-dispatch.patch) | Fewer small kernels per layer; faster output-head reads. |
| [Multi-stage sessions](patches/multi-stage-native-session-runtime.patch), [DFlash2 on a layer split](patches/layer-split-dflash-prompt-taps.patch) | Native session save/restore and the drafter work across both GPU stages. |
| [Automatic persistent prefixes](patches/automatic-persistent-prefix-cache.patch) | Restart-surviving reuse of shared prompt prefixes, with checkpoints, eviction and safe finalization. |
| [EPYC/NPS4 expert placement](patches/nps4-expert-row-placement.patch), [service lifecycle](patches/epyc-nps4-service-lifecycle.patch) | Expert rows and CPU workers stay on the same NUMA node. |
| [Bounded image keys](patches/bounded-image-wire-keys.patch) | Rejects malformed image identifiers before native parsing. |

Every patch and its activation are listed in [patches/README.md](patches/README.md). Several are
explicit rounding-changing operator choices, enabled by the profiles; whole requests are not bitwise
identical to upstream.

## Setup

### 1. Install prerequisites and clone

Install CUDA 13.4 and a compatible driver ([NVIDIA guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)), then:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build python3 python3-venv pkg-config

git clone --recurse-submodules https://github.com/airium/strata-airified.git
cd strata-airified
```

The submodules are pinned: **do not update them to upstream `main`**. `build.sh` applies the patches
and rejects unexpected revisions or conflicts.

### 2. Create the Python environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r vendor/strata/requirements.txt 'huggingface_hub[cli]'
```

### 3. Download the model, projector and drafter

Download only these files into the sibling `models/` directory (about 120 GB). Run `.venv/bin/hf auth login` first if needed.

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

The drafter pack is used as published. Keep all three shards together; the projector stays F16.

### 4. Build

```bash
bash build.sh          # the engine (incremental; --rebuild for a clean build)
bash build-vision.sh   # the image reader
```

Defaults: `CUDA_TOOLKIT=/usr/local/cuda-13.4`, `BUILD_DIR=build/sm75`, `BUILD_JOBS=16` (8 for the
reader). Stale CMake paths are refreshed automatically after a move; recreate `.venv` after moving
the checkout. `build/sm75/BUILD.json` records the source, patch and binary hashes.

### 5. Prepare the pack

Strata reads a few small projections as BF16 and serves everything else from the GGUF itself. Make the
compatibility pack once (about 1.5 GB; the experts are not copied):

```bash
STRATA_GGUF_PY="$PWD/vendor/llama.cpp/gguf-py" .venv/bin/python vendor/strata/tools/iq_pack.py \
  --gguf "$MODEL_ROOT/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/Qwen3.8-Flash-Next-Uncensored-Q4_K_M-00001-of-00003.gguf" \
  --out runtime/packs/orca-q4_k_m --compat-bf16
```

This also exports the model's tokenizer into the pack. See the upstream
[Orca workflow](vendor/strata/docs/ORCA.md) for background.

### 6. Select your GPUs and launch

```bash
test -f config.env || cp config.example.env config.env
```

**Edit `config.env` first.** Set `GPU_DEVICES` to your pair, usually `0,1` on a two-card host; the
template uses `2,3`. Model, pack and drafter paths match the steps above.

```bash
bash serve.sh --print-config
bash serve.sh                 # image reader from VISION_MODE in config.env (cpu in the template)
bash serve.sh --vision gpu    # this launch only: off, cpu or gpu
```

The server stays in the foreground at `http://127.0.0.1:3893/v1`; Ctrl-C stops it. Startup loads
about 75 GiB of experts into RAM and fills the VRAM expert caches, which takes a few minutes.
**Run it as your normal user, not with `sudo`** (see [Troubleshooting](#troubleshooting)).

## Configuration

Host settings live in the ignored `config.env` ([template](config.example.env)); explicit values
override the built-in defaults and the profile.

| Setting | Default / purpose |
| --- | --- |
| `PROFILE` | `q4km-int8kv-256k-dflash2-sm75-tcwide-c50-trim24-fused-hcnative-persistent` |
| `GPU_DEVICES` | `2,3` in the template; **set this to your pair** |
| `HOST` / `PORT` | `127.0.0.1` / `3893` |
| `SERVED_NAME` | `orcarouter-qwen3.8-flash-next-q4-k-m` |
| `MODEL_ROOT` | Sibling `../models`; `TARGET_MODEL` (first shard) and `MMPROJ` can override individual files |
| `PACK_DIR` / `DRAFT_PACK` | `runtime/packs/orca-q4_k_m` / `runtime/dflash2/…selq4.pack` |
| `PYTHON_BIN` | `.venv/bin/python` in the template |
| `CPU_NUMA_POLICY` | `auto`; `off` disables the EPYC/NPS4 placement |
| `PERSISTENT_CACHE_DIR` / `_GIB` / `_MIN_TOKENS` | Sibling `../strata-kv-caches` / `64` / `1024` |
| `VISION_MODE` / `VISION_MAX_TOKENS` | `cpu` in the template (`off`, `cpu` or `gpu`) / `auto` (1024 CPU, 4096 GPU) |
| `RUNTIME_DIR` | `runtime/`: resolved config, logs and lease records |

### Profiles

| Profile | Use |
| --- | --- |
| `…-trim24-fused-hcnative-persistent` (**default**) | Fixed 24/24 split with stage-local dense weights, native hyper-connection reads, SM75 fusions, persistent prefixes. Symmetric pair of 22 GiB cards. |
| `…-trim24-fused-hcnative-kqmmq-direct-persistent` (opt-in) | The default route with Q4_K expert prompt products through int8 MMQ (`STRATA_MMQ_KQ=1`) and cached experts read in place (`STRATA_MMQ_DIRECT=1`). Prefill about +12% (32K/128K, prose/code) with decode unchanged, but a measurable quality loss on the reference checks (next-token accuracy 73 → 71 of 100, gold NLL +0.014): choose it for prompt-heavy work where that is acceptable. |
| `…-sm75-tcwide-c50-persistent` (**fallback**) | Automatic layer split with the same SM75 operators and persistent prefixes. Other GPU pairs. |
| `…-trim24-fused-persistent` | The default route without native hyper-connection reads. |
| `…-sm75-tcwide-c50` | The fallback without persistent prefixes. |
| `q4km-int8kv-256k-dflash2` | Original operator baseline, DFlash confidence 0.35. |

All share the checkpoint, INT8 KV, 262144 cells and DFlash2. The SM75 operators change rounding, and
residency differs between profiles, so outputs are not bitwise identical across them. Requalify
quality, memory and speed after changing the checkpoint, toolkit or hardware.

## Persistent prefixes

The `-persistent` profiles save native snapshots at completed request boundaries and reuse the
longest compatible prefix on later requests, including after a restart. Only the uncached suffix is
prefilled. Snapshots hold INT8 KV, recurrent/indexer state, per-GPU checkpoints and the DFlash2
lineage; checkpoints every 16K tokens let a changed question resume at an earlier one.

- `PERSISTENT_CACHE_DIR` is created with private permissions and holds conversation content; keep it
  outside the repository. `prefixes/` is automatic; `sessions/` is for explicit slot save/restore.
- Committed snapshots share an LRU-like budget (`PERSISTENT_CACHE_GIB`); replacement needs temporary
  extra space. Saving and restoring add latency at request boundaries.
- Build, operator, model, KV, context and layer-layout changes select a different namespace; corrupt
  or incompatible files fall back to a normal read.
- `"cache_prompt": false` bypasses reuse and saving; `"strata_checkpoint": false` reuses without
  saving. Image-bearing requests skip the disk tier (repeated images still reuse the in-process cache).

Reuse is confirmed by the native counters in the response (`prompt_tokens_details.cached_tokens`), not
by file creation.

## Image input

`VISION_MODE=cpu` runs the F16 projector on the CPU (`VISION_THREADS`, default 16) and takes no GPU
memory. `VISION_MODE=gpu` encodes on `VISION_CUDA_DEVICE` (default: the first text GPU) instead. The
reader starts before the engine, so that card's expert cache simply gets smaller: on the default
route, with a 4096-token budget, the first card keeps 4988 of 5521 slots, prefill is unchanged and
decode is about 1–3.5% lower, while a 4000-token image encodes in about 6 s instead of over a minute.
Choose `cpu` for text-heavy work and `gpu` for frequent or large images; switching is a restart,
and `bash serve.sh --vision gpu` (or `cpu`, `off`) overrides `config.env` for one launch.

`VISION_MAX_TOKENS` is a per-image resolution budget, not a fixed count. `auto` (default) gives 1024
on the CPU and 4096 on a GPU; a number overrides it. Encoding time grows faster than the token count
(about 7 s for 1000 tokens and over a minute for 4000 on 16 CPU threads, under 6 s for 4000 on one
2080 Ti). Smaller images keep their own size. Image tokens count against the 262144-cell context. `/health` reports
`"images": true`; clients that define model capabilities manually must enable image input.

## Automatic EPYC/NPS4 CPU placement

With `CPU_NUMA_POLICY=auto`, the launcher checks for one AMD EPYC socket with four balanced
CPU-and-memory NUMA nodes inside the service's cgroup limits and an NPS4-capable build. Only then does
it assign expert rows and CPU workers per node. Intel, Ryzen, dual-socket, other node counts and
restricted cgroups keep the default route; `--print-config` shows the decision under `cpu_numa_policy`.

If a `numad` daemon is running, only the engine process is excluded from it, before model loading,
and a watcher removes the exclusion when the engine exits. This uses `sudo -n` for numad's client, so
the user needs passwordless sudo for it; otherwise set `CPU_NUMA_POLICY=off`. Without numad nothing is
needed. Changes take effect at the next start.

## API

OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:3893/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"orcarouter-qwen3.8-flash-next-q4-k-m","messages":[{"role":"user","content":"Explain binary search briefly."}],"max_tokens":256,"temperature":0}'
```

Images use `image_url` content parts (data URLs). `"reasoning_effort":"none"` disables thinking for a
request. Endpoints: `/health`, `/v1/models`, `/v1/chat/completions`, `/slots`,
`/metrics`.

## Validation

Passed on the reference host with the default profile:

- **Full window:** 261112 input + 1024 natural output tokens filling the 262136 usable cells.
- **Full-window image:** a 261112-token image prompt read correctly; a repeated image reused 261105
  tokens; a changed image was detected (245760 reused, new label read).
- **Restart reuse:** a fresh process reused 4926 (prose) and 5641 (code) prefix tokens with native
  counters matching the API; opt-out, manual save/restore and image input also passed.
- **Quality:** 31 functional tasks and 8 executed generated functions; teacher-forced next-token loss
  over 12295 tokens in 25 documents within +0.01 nats of the previous route (+0.0008, 95% interval
  −0.0028 to +0.0043).
- Wrapper tests: `.venv/bin/python -m unittest discover -s tests`.

These are functional, capacity and finite quality checks on one host, not a broad model evaluation
or a guarantee for other hardware or checkpoints.

## Troubleshooting

- **Do not use `sudo`.** Nothing in the server needs root. A root run creates root-owned cache and
  lease files that later normal runs cannot reuse or clean up, and runs the model server as root.
- **Out of VRAM at startup:** another process is using the GPUs, or the cards have less than 22 GiB;
  use the fallback profile and check `--print-config`.
- **Out of RAM:** the experts alone take about 75 GiB; close other large processes.
- **`numad exclusion requires noninteractive permission`:** set `CPU_NUMA_POLICY=off`, or allow
  passwordless `sudo` for numad's client.
- The startup log reports the expert-cache sizes, layer ranges and, for the default profile,
  `STRATA_HC_NATIVE=1: … read natively`.

## Development

Patch workflow, experiment worktrees, tests and diagnostics: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Upstream and model attribution

Wrapper code and patches: [MIT](LICENSE). [Strata](https://github.com/Niko1221/Strata) (MIT) and
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) keep their licenses in the submodules;
[upstream.json](upstream.json) records the pins and patch order. Download the weights separately from
OrcaRouter and the drafter from the Strata release; their pages specify licensing. Results apply to
the named checkpoint and settings, not to every Qwen variant.
