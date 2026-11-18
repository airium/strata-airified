import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorktreeLaunch(unittest.TestCase):
    def test_server_module_and_resource_paths_follow_selected_worktree(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "selected-source"
            (source / "serve").mkdir(parents=True)
            (source / "serve/__init__.py").touch()
            (source / "serve/server.py").write_text(
                "import json,os,sys\n"
                "config=json.load(open(sys.argv[sys.argv.index('--config')+1]))\n"
                "print(json.dumps({'module':__file__,'cwd':os.getcwd(),'config':config}))\n")
            for name in ("native", "target.gguf", "draft.pack"):
                (folder / name).touch()
            (folder / "pack/tokenizer").mkdir(parents=True)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("STRATA_", "VISION_"))}
            env.update(CONFIG_FILE="/dev/null", PYTHON_BIN=sys.executable,
                       UPSTREAM_MANIFEST=str(ROOT / "upstream.json"), STRATA_SOURCE_DIR=str(source),
                       RUNTIME_DIR=str(folder / "runtime"), SERVER_BIN=str(folder / "native"),
                       TARGET_MODEL=str(folder / "target.gguf"), PACK_DIR=str(folder / "pack"),
                       DRAFT_PACK=str(folder / "draft.pack"), ENGINE_LIBRARY_PATH=str(folder), VISION_MODE="off")
            done = subprocess.run(["bash", str(ROOT / "serve.sh")], env=env, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            result = json.loads(done.stdout)
            self.assertEqual(Path(result["module"]), source / "serve/server.py")
            self.assertEqual(Path(result["cwd"]), source)
            cfg = result["config"]
            self.assertEqual(Path(cfg["cwd"]), source)
            self.assertEqual(cfg["args"][cfg["args"].index("--expert-profile") + 1],
                             str(source / "data/expert-profile.bin"))


if __name__ == "__main__":
    unittest.main()
