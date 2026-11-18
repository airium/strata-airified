#!/usr/bin/env python3
"""Apply manifest patches without resetting or staging the vendor worktree."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from source_config import load_manifest, rooted, source_dir

ROOT = Path(__file__).resolve().parents[1]


def git(source, args, **kwargs):
    return subprocess.run(["git", "-C", str(source), *args], **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="diagnostic build only; never reset local edits")
    parser.add_argument("--manifest", help="manifest file; defaults to UPSTREAM_MANIFEST or upstream.json")
    parser.add_argument("--source", help="source checkout; defaults to STRATA_SOURCE_DIR or the manifest path")
    options = parser.parse_args()
    _, manifest = load_manifest(options.manifest)
    selected_source = source_dir(manifest, options.source)
    for entry in [manifest] + manifest.get("dependencies", []):
        source = selected_source if entry is manifest else rooted(entry["path"])
        result = git(source, ["rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != entry["revision"]:
            raise SystemExit(f"Unexpected revision in {entry['path']}: expected {entry['revision']}; left unchanged")
    for entry in manifest.get("dependencies", []):
        source = ROOT / entry["path"]
        if git(source, ["diff", "--quiet", "HEAD", "--"]).returncode:
            raise SystemExit(f"Undeclared dependency edits in {entry['path']}; preserved.")
    source = selected_source
    patches = [rooted(p) for p in manifest["patches"]]
    # Build the declared series in an isolated index. Compare complete prefix
    # trees, not each patch's reverse applicability: later patches can edit the
    # same hunk and legitimately make earlier reverse checks fail.
    with tempfile.TemporaryDirectory(prefix="strata-patch-index-") as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))
        git(source, ["read-tree", manifest["revision"]], env=env, check=True)
        trees = []
        matching_prefix = None
        for count in range(len(patches) + 1):
            if count:
                git(source, ["apply", "--cached", str(patches[count - 1])], env=env, check=True)
            tree = git(source, ["write-tree"], env=env, capture_output=True, text=True, check=True).stdout.strip()
            trees.append(tree)
            git(source, ["update-index", "--refresh"], env=env, capture_output=True)
            if git(source, ["diff-files", "--quiet"], env=env).returncode == 0:
                matching_prefix = count
        if matching_prefix == len(patches):
            print("Declared patch series already applied.")
        elif matching_prefix is not None and not options.check:
            diff = git(source, ["diff", "--binary", trees[matching_prefix], trees[-1]], capture_output=True, check=True).stdout
            check = git(source, ["apply", "--check"], input=diff, capture_output=True)
            if check.returncode:
                raise SystemExit("Patch conflict; local edits preserved: " + check.stderr.decode())
            git(source, ["apply"], input=diff, check=True)
            print(f"Applied patches {matching_prefix + 1} through {len(patches)}.")
        elif options.allow_dirty:
            diff = git(source, ["diff", "--binary", manifest["revision"], trees[-1]], capture_output=True, check=True).stdout
            reverse = git(source, ["apply", "--reverse", "--check"], input=diff, capture_output=True) if diff else None
            if reverse is not None and reverse.returncode:
                raise SystemExit("Cannot verify declared patches among local edits; preserved.")
            print("Diagnostic override: declared patches present; other edits preserved.")
        else:
            raise SystemExit("Vendor differs from the declared patch set; preserved. Review edits or use --allow-dirty for a diagnostic build.")
    print("Pinned source and patch set verified.")


if __name__ == "__main__":
    main()
