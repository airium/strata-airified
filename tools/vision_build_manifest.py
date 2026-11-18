#!/usr/bin/env python3
"""Record standalone vision-reader provenance without changing the text build."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
build = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "upstream.json").read_text())
def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
cache = {}
for line in (build / "CMakeCache.txt").read_text().splitlines():
    if "=" in line and ":" in line.split("=", 1)[0]:
        key, value = line.split("=", 1)
        cache[key.split(":", 1)[0]] = value
inputs = ["CMakeLists.txt", "strata_vision.cpp", "gguf_meta.h", "gpu_arch.h"]
binary = build / "bin/strata-vision"
record = {"source_revision": manifest["revision"], "dependencies": manifest["dependencies"],
          "source_files": {name: sha(root / "vendor/strata/tools/vision" / name) for name in inputs},
          "cmake": {key: value for key, value in cache.items() if key.startswith(("STRATA_", "GGML_")) or key in
                    ["CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER", "CMAKE_CUDA_ARCHITECTURES", "LLAMA_DIR"]},
          "payload": {"path": "bin/strata-vision", "bytes": binary.stat().st_size, "sha256": sha(binary)}}
(build / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
print(build / "BUILD.json")
