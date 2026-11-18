#!/usr/bin/env python3
"""Export selected vendor edits relative to the declared patch series, without staging."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from source_config import load_manifest, rooted, source_dir

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("name", help="new patch filename, without directory")
parser.add_argument("paths", nargs="+", help="changed paths relative to vendor/strata")
parser.add_argument("--manifest", help="manifest file; defaults to UPSTREAM_MANIFEST or upstream.json")
parser.add_argument("--source", help="source checkout; defaults to STRATA_SOURCE_DIR or the manifest path")
parser.add_argument("--patch-dir", help="output directory; defaults to STRATA_PATCH_DIR or patches/")
args = parser.parse_args()
if Path(args.name).name != args.name or not args.name.endswith(".patch"):
    raise SystemExit("Use a plain .patch filename")
manifest_path, manifest = load_manifest(args.manifest)
source = source_dir(manifest, args.source)
def git(argv, **kw):
    return subprocess.run(["git", "-C", str(source), *argv], **kw)
actual = git(["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
if actual != manifest["revision"]:
    raise SystemExit("Unexpected source revision; no patch written")
target = rooted(args.patch_dir or os.environ.get("STRATA_PATCH_DIR") or "patches") / args.name
if target.exists():
    raise SystemExit("Patch already exists; review it instead of overwriting")
for p in args.paths:
    if Path(p).is_absolute() or ".." in Path(p).parts:
        raise SystemExit("Paths must stay inside the vendor tree")
with tempfile.TemporaryDirectory(prefix="strata-export-index-") as temp:
    env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))
    git(["read-tree", manifest["revision"]], env=env, check=True)
    for p in manifest["patches"]:
        git(["apply", "--cached", str(ROOT / p)], env=env, check=True)
    for p in args.paths:
        if git(["ls-files", "--error-unmatch", "--", p], env=env, capture_output=True).returncode:
            git(["add", "--intent-to-add", "--", p], env=env, check=True)
    diff = git(["diff", "--binary", "--", *args.paths], env=env, capture_output=True, check=True).stdout
if not diff:
    raise SystemExit("No undeclared edits in those paths")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(diff)
name = str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target)
manifest["patches"].append(name)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(name)
