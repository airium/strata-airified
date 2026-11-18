# Local patch series

`upstream.json` records application order and the exact base. Apply/check with
`python3 tools/apply_patches.py [--check]`; export focused vendor edits with
`python3 tools/record_patch.py NAME.patch PATH...`. Neither helper stages the
vendor's real index. Unexpected revisions, conflicts and undeclared tracked edits
are preserved and reported.

| Patch | Purpose | Runtime activation |
| --- | --- | --- |
| `q5_0-mmq-coverage` | Independent reference coverage for Q5_0 expert down-products. | Tests; MMQ itself remains an upstream opt-in. |
| `bf16-tc-direct-fp16-input` | Avoid unused conversion scratch and slicing for an existing contiguous FP16 image. | `STRATA_BF16_TC=1 STRATA_BF16_TC_WIDE_X=1`; explicit rounding-changing route. |
| `bf16-tc-operator-scope` | Isolate hyperconnection/up projections while preserving other BF16 products. | `STRATA_BF16_TC_SCOPE=hc` or `up`, with TC enabled; default `all`. |
| `q5_0-grouped-expert-coverage` | Add Q5_0 down weights to grouped K-quant experts; make the old-kernel parity reference bypass V2K. | `STRATA_EXPERT_V2K=1 STRATA_EXPERT_V2K_Q50=1`. |
| `bf16-fp32-sgemm-experiment` | Exact BF16-to-FP32 widening followed by SGEMM on Turing. | `STRATA_BF16_F32=1`; experimental accumulation order. |
| `stable-cmake-overlay` | Keep generated-header timestamps stable on no-op configure. | Build-only; no runtime arithmetic change. |
| `bf16-fp32-scope-and-alias-guard` | Avoid recursive scratch aliasing; select HC-up or exclude the slower down/single-row shapes; expand parity cases. | `STRATA_BF16_F32_SCOPE=up` or `fast` with FP32 enabled. |
| `kquant-grouped-gu-two-rows` | Optional two-row gate/up layout and Q4_K/Q5_0,Q8_0 operator benchmarks. | `STRATA_EXPERT_V2K_GU_ROWS=2`; default four rows is retained. |
| `multi-stage-session-codec` | Bounded v2 stage bundles and opaque draft attachment; legacy v1 golden preserved. | Native disk sessions. |
| `nullable-stage-snapshot-and-dflash-log` | Stage-local snapshots without a fake MTP layer; portable DFlash quantized ring lineage. | Native disk sessions with DFlash2. |
| `multi-stage-native-session-runtime` | Per-device capture/restore, full prevalidation, build/arithmetic/layout identity and explicit reuse opt-out. | `SAVE`/`RESTORE`; optional diagnostic `STRATA_SESSION_VERIFY=1`. |
| `layer-split-dflash-prompt-taps` | Deliver intermediate-stage prompt taps before the pipeline handoff. | DFlash2 layer split; fixes an upstream missing-context path. |
| `automatic-persistent-prefix-cache` | Atomic bounded native snapshots with token-prefix selection and fresh-process restore. | `-persistent` profile / `STRATA_PERSISTENT_CACHE_DIR`. |
| `multi-stage-streaming-tests` | Streamed/captured byte equality, later-stage publication failure and build identity coverage. | Host tests. |
| `dflash-persistent-ring-parity-test` | Fresh-drafter restored K/V matches uninterrupted ring bytes. | `dflash2_ring_test`. |
| `persistent-metadata-eviction-guard` | Ignore malformed metadata during owned-cache eviction. | Automatic cache maintenance. |
| `retain-periodic-prefix-checkpoints` | Preserve the bounded checkpoint chain for changed-question/branch reuse; account for its save-time RAM. | `STRATA_SESSION_ALL_CHECKPOINTS=1` in the persistent profile only. |
| `persistent-request-boundary-finalization` | Save after native DONE even when HTTP closes generation at EOS; invalidate the live hint after manual restore. | Persistent request lifecycle. |
| `bounded-image-wire-keys` | Reject overflowing image IDs before the native uint64 parser; invalid internal keys use the native embedding hash instead. | Image-bearing native requests. |
| `hc-native-blocks` | Decode hyper-connection read from the GGUF's own Q4_K/Q6_K down/inject and Q5_0 up blocks (FP32 activations, split-K down with a fixed-order reduction). | `STRATA_HC_NATIVE=1`; changes decode values/summation order. |
| `hc-native-no-bf16` | Skip the BF16 hyper-connection down/up rows when native copies serve decode; the prompt/single-token paths rebuild BF16 with the pack's exact values. | `STRATA_HC_NATIVE=1 STRATA_HC_NATIVE_NO_BF16=1`; residency change only. |
| `hc-native-session-identity` | Native manual-session signatures include an explicit `STRATA_HC_NATIVE`. | Native session save/restore. |
| `hc-native-load-message` | The native hyper-connection load line reports how many projections run without BF16 rows. | Startup log only. |
| `mmq-tab-kquant-types` | `STRATA_MMQ_TAB_TYPES` selects the pointer-table prompt MMQ types (default: upstream's nine); `build.sh` builds `q4_K;q5_0;q8_0`, this pack's expert types. `prefill_mmq_tab_test` derives its cases from the built types, sets K-quant fp16 scales correctly and runs on sm_75 (555/555 bitwise there). | `STRATA_MMQ_DIRECT=1` with `STRATA_MMQ_KQ=1` (the `-kqmmq-direct` profile); otherwise only the kernels' VRAM. |

A compiled opt-in is not a qualified deployment default. Keep arithmetic-changing
paths off unless separately validated for the checkpoint, GPU and library build.
Host-specific measurements and rejected/selected experiments belong in the
workspace archives, not this portable source tree. Only CUDA SM75 has been built
and tested locally; this is not multi-backend qualification.

Additional optional host diagnostics use host-only NVTX ranges, a CPU launch timer,
and an off-host persistent graph-submission worker. The worker is off in normal
profiles. SM75-specific CUDA dispatch exposes the existing GDN conv/alpha-beta and
MoE hit-add/combine fusions plus a shape-limited Q6_K output-head prefetch path.
Native manual-session signatures include explicitly selected operator modes even
when automatic Python persistence is disabled. Rejected kernel/precision probes
remain outside the deployment patch series.
