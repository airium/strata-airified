#!/usr/bin/env python3
"""Record the exact local build inputs and payload hashes beside the engine."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from source_config import load_manifest, source_dir

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--build-dir', type=Path, required=True)
a = p.parse_args()
build = a.build_dir.resolve()
manifest_path, upstream = load_manifest()
source = source_dir(upstream)
cache = {}
for line in (build / 'CMakeCache.txt').read_text().splitlines():
    if '=' in line and ':' in line.split('=', 1)[0]:
        key, value = line.split('=', 1)
        cache[key.split(':', 1)[0]] = value
if Path(cache['CMAKE_HOME_DIRECTORY']).resolve() != source:
    raise SystemExit('Build source differs from the selected source; no provenance written')
version = re.search(r'project\(strata VERSION ([\d.]+)', (source / 'CMakeLists.txt').read_text()).group(1)
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
metadata = {'version': version, 'source_revision': upstream['revision'],
            'source_dir': str(source), 'manifest_file': str(manifest_path),
            'manifest_sha256': sha(manifest_path),
            'dependencies': upstream.get('dependencies', []),
            'patches': [{'path': name, 'sha256': sha(ROOT / name)} for name in upstream['patches']],
            'cmake': {k: v for k, v in cache.items() if k.startswith('STRATA_') or k in
                      ['CMAKE_BUILD_TYPE','CMAKE_CXX_COMPILER','CMAKE_CUDA_COMPILER','CMAKE_CUDA_ARCHITECTURES']},
            'payloads': {name: {'bytes': (build / name).stat().st_size, 'sha256': sha(build / name)}
                         for name in ['strata','libstrata_kq.so']}}
(build / 'BUILD.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(build / 'BUILD.json')
