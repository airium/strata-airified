"""Shared, explicit source/manifest selection for deployment and worktree builds.

Manifest paths remain relative to the wrapper root; absolute paths are allowed
for private experiment manifests. Selecting another source never moves a checkout.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rooted(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_manifest(value=None):
    path = rooted(value or os.environ.get("UPSTREAM_MANIFEST") or "upstream.json")
    return path, json.loads(path.read_text())


def source_dir(manifest, override=None):
    return rooted(override or os.environ.get("STRATA_SOURCE_DIR") or manifest["path"])


if __name__ == "__main__":
    _, manifest = load_manifest()
    print(source_dir(manifest))
