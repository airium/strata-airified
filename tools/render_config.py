#!/usr/bin/env python3
"""Resolve one portable profile plus explicit host environment into Strata JSON."""
import json
import importlib.util
import os
from pathlib import Path
import sys
from source_config import load_manifest, source_dir

ROOT = Path(__file__).resolve().parents[1]
_, selected_manifest = load_manifest()
SOURCE = source_dir(selected_manifest)


def path(name, default):
    p = Path(os.environ.get(name) or default).expanduser()
    return str((ROOT / p).resolve() if not p.is_absolute() else p)


def integer(name, default, minimum=1):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        raise SystemExit(f"{name} must be an integer >= {minimum}") from None
    if value < minimum:
        raise SystemExit(f"{name} must be an integer >= {minimum}")
    return value


profile_name = os.environ.get("PROFILE", "q4km-int8kv-256k-dflash2-sm75-tcwide-c50")
if "/" in profile_name or "\\" in profile_name:
    raise SystemExit("PROFILE must name a file in profiles/")
profile = json.loads((ROOT / "profiles" / (profile_name + ".json")).read_text())
model_root = Path(path("MODEL_ROOT", "../models"))
first = path("TARGET_MODEL", str(model_root / "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/Qwen3.8-Flash-Next-Uncensored-Q4_K_M-00001-of-00003.gguf"))
pack = path("PACK_DIR", "runtime/packs/orca-q4_k_m")
draft = path("DRAFT_PACK", "runtime/dflash2/qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack")
engine = path("SERVER_BIN", "build/sm75/strata")
runtime = Path(path("RUNTIME_DIR", "runtime"))
runtime.mkdir(parents=True, exist_ok=True)
env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "LD_LIBRARY_PATH": path("ENGINE_LIBRARY_PATH", str(Path(engine).parent))}
env.update(profile.get("environment", {}))
for key in ["STRATA_SM75_PERF", "STRATA_BF16_TC", "STRATA_BF16_TC_SCOPE",
            "STRATA_BF16_TC_WIDE_X", "STRATA_BF16_F32", "STRATA_BF16_F32_SCOPE", "STRATA_XN_F16",
            "STRATA_MMQ_KQ", "STRATA_MMQ_DIRECT", "STRATA_EXPERT_V2K", "STRATA_EXPERT_V2K_Q50",
            "STRATA_PF_HC_REGION", "STRATA_TSUM", "STRATA_PA_V2T", "STRATA_AUX_CPUS", "STRATA_MANIFEST",
            "STRATA_NUMA_SPLIT", "STRATA_POOL_SUMMARY",
            "STRATA_FUSE_CONVAB", "STRATA_FUSE_HITS", "STRATA_SM75_HEAD_PF",
            "STRATA_HC_NATIVE", "STRATA_HC_NATIVE_NO_BF16"]:
    if key in os.environ:
        env[key] = os.environ[key]
args = ["--pack", pack, "--native", first, "--ple-gguf", path("PLE_GGUF", first),
        "--expert-profile", str(SOURCE / "data/expert-profile.bin"), "--expert-cache", "auto",
        "--prefill", str(profile["prefill"]), "--spec", str(profile["spec"]),
        "--spec-min-p", str(profile["spec_min_p"]), "--max-context", str(profile["context"]),
        "--kv", profile["kv"], "--prompt-cache", str(profile["prompt_cache"]),
        "--prompt-cache-every", str(profile.get("prompt_cache_every", 0)), "--conversation-cache-mib", "0", "--draft", "dflash2",
        "--draft-model", draft, "--draft-fallback", "none", "--mtp-draft-vocab", str(SOURCE / "data/draft_vocab.bin"),
        "--draft-conf", str(profile["draft_conf"])]
trim_weights = profile.get("trim_stage_weights", False)
if not isinstance(trim_weights, bool):
    raise SystemExit("profile trim_stage_weights must be a boolean")
if trim_weights:
    args.append("--trim-stage-weights")
if profile.get("prompt_cache_tail"):
    args.append("--prompt-cache-tail")
if profile.get("persistent_cache"):
    cache_root = path("PERSISTENT_CACHE_DIR", "../strata-kv-caches")
    env.update(STRATA_PERSISTENT_CACHE_DIR=cache_root,
               STRATA_PERSISTENT_CACHE_GIB=os.environ.get("PERSISTENT_CACHE_GIB", "64"),
               STRATA_PERSISTENT_CACHE_MIN_TOKENS=os.environ.get("PERSISTENT_CACHE_MIN_TOKENS", "1024"))
cfg = {"exe":engine,"args":args,"cwd":str(SOURCE),"tokenizer":str(Path(pack)/"tokenizer"),
       "model_name":os.environ.get("SERVED_NAME","orcarouter-qwen3.8-flash-next-q4-k-m"),
       "gpu":[int(x) for x in os.environ.get("GPU_DEVICES","2,3").split(",")],"gpu_order":"as_given",
       "layer_split":profile["layer_split"],"parallel":1,"host":os.environ.get("HOST","127.0.0.1"),
       "port":int(os.environ.get("PORT","3893")),"fit_max_tokens":False,"env":env,
       "log":str(runtime/(profile_name+".engine.log"))}
cpu_mode = os.environ.get("CPU_NUMA_POLICY", "auto").lower()
if cpu_mode not in ("auto", "off"):
    raise SystemExit("CPU_NUMA_POLICY must be auto or off")
cpu_module = SOURCE / "serve/numa_lifecycle.py"
env["AIRIFIED_NPS4_GUARD"] = "0"
if cpu_module.is_file():
    spec = importlib.util.spec_from_file_location("airified_cpu_policy", cpu_module)
    cpu_policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(cpu_policy)
    report = cpu_policy.configuration(engine, env, cpu_mode, str(runtime / "numa-lifecycle"))
    if report["enabled"]:
        env.update(STRATA_NUMA_SPLIT="4", AIRIFIED_NPS4_GUARD="1",
                   AIRIFIED_NPS4_STATE_DIR=str(runtime / "numa-lifecycle"))
    cfg["cpu_numa_policy"] = report
else:
    cfg["cpu_numa_policy"] = {"mode":cpu_mode,"enabled":False,"reason":"selected source lacks managed NUMA lifecycle support"}
if profile.get("persistent_cache"):
    cfg["slot_save_path"] = str(Path(cache_root) / "sessions")
vision_mode = os.environ.get("VISION_MODE", "off").lower()
if vision_mode not in ("off", "cpu", "gpu"):
    raise SystemExit("VISION_MODE must be off, cpu or gpu")
if vision_mode != "off":
    vision_bin = path("VISION_BIN", "build/vision-sm75/bin/strata-vision")
    projector = path("MMPROJ", str(Path(first).parent / "mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"))
    # auto: the per-image budget follows the reader's device (CPU encoding cost grows steeply with size)
    cap = os.environ.get("VISION_MAX_TOKENS", "auto").strip().lower() or "auto"
    limit = (4096 if vision_mode == "gpu" else 1024) if cap == "auto" else integer("VISION_MAX_TOKENS", 1024)
    threads = integer("VISION_THREADS", 16)
    cfg["vision"] = {"exe": vision_bin, "mmproj": projector, "model": first,
                     "gpu": vision_mode == "gpu", "max_tokens": limit, "threads": threads,
                     "elastic": False}
    args.append("--vision")
    toolkit = Path(path("CUDA_TOOLKIT", "/usr/local/cuda-13.4"))
    libraries = path("VISION_LIBRARY_PATH", str(toolkit / "lib64"))
    env["LD_LIBRARY_PATH"] += os.pathsep + libraries
    if vision_mode == "gpu":
        cfg["vision"]["cuda_device"] = integer("VISION_CUDA_DEVICE", cfg["gpu"][0], minimum=0)
        cfg["vision"]["cpu_exe"] = vision_bin
        reserve = integer("VISION_VRAM_RESERVE_MIB", 700)
        args.extend(["--vram-reserve-mib", str(reserve)])
if "--check" in sys.argv:
    for p in [engine,first,pack,draft,cfg["tokenizer"],*env["LD_LIBRARY_PATH"].split(os.pathsep),str(SOURCE/"serve/server.py")]:
        if not Path(p).exists():raise SystemExit("Missing runtime input: "+p)
    if vision_mode != "off":
        for p in [vision_bin, projector, libraries]:
            if not Path(p).exists():raise SystemExit("Missing vision input: "+p)
print(json.dumps(cfg,indent=2))
