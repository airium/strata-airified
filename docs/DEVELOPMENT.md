# Development

How this wrapper changes the pinned engine, and how to run engine experiments without disturbing a
deployment. Deployment itself is in the [README](../README.md).

## Repository model

- `vendor/strata/` and `vendor/llama.cpp/` are pinned submodules. Do not advance them automatically
  (`git submodule update --init --recursive`, never `--remote`).
- Local engine changes are patch files in `patches/`, applied in the order `upstream.json` lists, on
  top of its exact `revision`. [patches/README.md](../patches/README.md) describes each one.
- `build.sh` verifies the revisions and applies only the declared series. Conflicting or undeclared
  edits are preserved and reported, never reset. `build/sm75/BUILD.json` records source, patch and
  payload hashes.
- Upstream's own agent instructions remain in `vendor/strata/AGENTS.md`.

## Changing the engine

Make focused edits in the vendor checkout (or an experiment worktree, below), then export them:

```bash
python3 tools/record_patch.py NAME.patch PATH...      # relative to the current series; does not stage
python3 tools/apply_patches.py --check                # the declared series matches the source tree
```

`record_patch.py` appends the new file to the manifest it was given. Both helpers accept
`--manifest` and `--source`; `record_patch.py --patch-dir` writes elsewhere without touching the
default manifest.

Keep arithmetic-changing paths behind explicit environment opt-ins until they are qualified, and
document their activation in `patches/README.md`. Do not commit weights, packs, caches, logs or
benchmark evidence.

## Worktree experiments

Keep a clean upstream clone as the base for linked experiment worktrees; leave the deployment
submodule pinned while they run. Reuse the guarded build recipe with explicit inputs:

```bash
UPSTREAM_MANIFEST=/path/to/experiment/upstream.json \
STRATA_SOURCE_DIR=/path/to/strata-worktree \
BUILD_DIR=/path/to/experiment/build \
bash build.sh
```

Extra engine CMake options (for example a build-time feature such as `-DSTRATA_MMQ_TAB=ON`) go in
`CMAKE_EXTRA_ARGS`, whitespace-separated, together with a separate `BUILD_DIR`.

The manifest must declare that exact revision and patch series. Manifest paths are relative to this
wrapper; private experiment manifests may use absolute paths. Never reuse another worktree's CMake
build directory, and never build an experiment by switching the live vendor checkout underneath a
running server.

To serve an experiment, put the same source/manifest overrides, a matching `SERVER_BIN` and
`ENGINE_LIBRARY_PATH`, and a separate `RUNTIME_DIR`, port and `PERSISTENT_CACHE_DIR` in an experiment
config, then run `CONFIG_FILE=/path/to/experiment.env bash serve.sh`. The source override selects both
the build inputs and the Python server, not just the native binary. The engine build selects the
`flash_next` model geometry.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

The wrapper tests cover profile rendering, patch workflow, source launch, persistent-prefix and
streaming behaviour, vision configuration and the NUMA lifecycle. They do not load a model.

## Diagnostics

- `STRATA_VERIFY_HOST_PROFILE=1`: host-only NVTX ranges around the decode window.
- `STRATA_GRAPH_SUBMIT_TIMING=1`: times the CUDA graph launch call without GPU stamps.
- `STRATA_VERIFY_LAUNCH_WORKER=1`: an off-host persistent graph-submission thread. Useful when Nsight
  node tracing delays submission; off for normal serving (a faster trace is not a production speedup).
- `STRATA_POOL_SUMMARY=1`: request-level CPU expert-row phase times and logical bytes (not DRAM bandwidth).

Profile with the deployed settings and compare against unprofiled runs; `STRATA_VERIFY_PROFILE` and
similar stamp modes change dispatch and are not production-representative.
