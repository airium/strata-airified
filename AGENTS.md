# Airified Strata contracts

Portable rules for this repository. Host-specific deployment preferences belong in the
operator's own workspace instructions (for example a parent directory's `AGENTS.md`).

- Keep upstream in pinned `vendor/strata`; local engine changes are reviewed patch
  files listed in `upstream.json`, not commits in the vendor history. The pinned
  llama.cpp/ggml dependency is `vendor/llama.cpp`.
- Build and patch helpers must reject unexpected revisions or conflicts without
  resetting, cleaning or staging unrelated vendor edits. Preserve explicit host
  overrides, checkpoints and caches. Refresh stale generated CMake paths after moves.
- Worktree experiments select `STRATA_SOURCE_DIR` and `UPSTREAM_MANIFEST` explicitly;
  normal deployment defaults remain the pinned vendor tree. Keep source/module,
  build/library and runtime/cache paths consistent, and never build an experiment
  by switching the live vendor checkout underneath its server.
- Use Conventional Commits and the configured signing policy for authorized
  commits. Do not commit or push without explicit user authorization.
- `README.md` and `README.zh-CN.md` are synchronized deployment-first public guides;
  developer workflow lives in `docs/DEVELOPMENT.md`. Keep personal paths, hosts and
  private network setup in ignored `config.env`, not public examples.
- Do not commit weights, packs, caches, logs or benchmark evidence to this repository.
- Do not run the server as root. The numad integration only needs noninteractive
  `sudo -n` for its PID-control client; root runs leave root-owned cache/lease files.
- CPU placement defaults to automatic only for a verified single-socket Linux
  EPYC/four-node topology and matching NPS4 build. Keep native-PID numad leases
  before exec, cgroup-bounded eligibility and identity-checked exit cleanup together.
  Do not exclude the API/vision processes or change daemon-wide policy. Other
  topologies keep their existing route; `CPU_NUMA_POLICY=off` disables automation.
- Persistent prefixes use the native versioned session codec, including every
  stage's recurrent/indexer state and DFlash lineage. Keep payloads outside source.
  File creation/restore alone is not a hit: verify native reused/read counts across
  fresh processes. Preserve request opt-out and build/layout isolation. Finalize
  snapshots only after native DONE, including when HTTP closes at EOS.
- Keep vision-reader placement explicit and separate from the text GPU pair. A
  shared GPU reader must load and warm up before auto-sizing the expert cache;
  do not assume layer-split caches support dynamic VRAM lending. Preserve image
  content-key matching. Automatic disk snapshots remain text-only until image
  identity and encoder compatibility are qualified for restart reuse.
- Do not promote an experimental arithmetic path solely on speed. Separate
  operator/logit checks, functional checks and full-window execution from configured
  capacity or health-ready. Treat profiled rates separately from headline rates.
- Fused q8_1 producers must preserve the actual consumer's division/rounding
  contract. CUDA dense and routed-expert quantizers are not universally byte-
  equivalent; validate both contracts rather than only the dense reference.
- Native hyper-connection reads must use the checkpoint's own block values and FP32
  activations. Drop BF16 rows only where a native copy exists; any BF16 rebuild must
  reproduce the pack's bytes exactly (gguf-py order, round-to-nearest-even).
- The fixed 24/24 stage-local profiles are specific to the qualified symmetric GPU
  pair. Keep the automatic-split profile available; do not silently apply a fixed
  split to other model geometries or hardware.
