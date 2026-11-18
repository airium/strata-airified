import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


class PatchWorkflow(unittest.TestCase):
    def test_export_apply_isolation_and_revision_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "vendor/strata"
            source.mkdir(parents=True)
            (root / "tools").mkdir()
            (root / "patches").mkdir()
            for tool in ["apply_patches.py", "record_patch.py", "source_config.py"]:
                shutil.copy2(TOOLS / tool, root / "tools" / tool)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.STDOUT)
            git("init", "-q")
            git("config", "user.name", "Patch test")
            git("config", "user.email", "patch-test@example.invalid")
            git("config", "commit.gpgsign", "false")
            (source / "engine.txt").write_text("base\n")
            (source / "other.txt").write_text("untouched\n")
            git("add", ".")
            git("commit", "-qm", "fixture")
            revision = git("rev-parse", "HEAD").decode().strip()
            (root / "upstream.json").write_text(json.dumps({"path":"vendor/strata","revision":revision,"patches":[]}))
            index_before = (source / ".git/index").read_bytes()
            (source / "engine.txt").write_text("optimized\n")
            subprocess.run([sys.executable,str(root/"tools/record_patch.py"),"engine.patch","engine.txt"],check=True,capture_output=True)
            self.assertEqual(index_before,(source/".git/index").read_bytes())
            check = [sys.executable,str(root/"tools/apply_patches.py"),"--check"]
            subprocess.run(check,check=True,capture_output=True)
            subprocess.run(check,check=True,capture_output=True)
            # Later patches may replace the exact hunk of earlier patches.
            (source / "engine.txt").write_text("optimized twice\n")
            subprocess.run([sys.executable,str(root/"tools/record_patch.py"),"second.patch","engine.txt"],check=True,capture_output=True)
            subprocess.run(check,check=True,capture_output=True)
            self.assertEqual(index_before,(source/".git/index").read_bytes())
            # Reapply the remaining series from a clean declared prefix.
            (source / "engine.txt").write_text("optimized\n")
            subprocess.run(check[:-1],check=True,capture_output=True)
            self.assertEqual((source/"engine.txt").read_text(),"optimized twice\n")

            # And from the untouched upstream tree.
            (source / "engine.txt").write_text("base\n")
            subprocess.run(check[:-1],check=True,capture_output=True)
            self.assertEqual((source/"engine.txt").read_text(),"optimized twice\n")
            (source / "other.txt").write_text("user edit\n")
            git("add","other.txt")
            staged = git("diff","--cached")
            self.assertNotEqual(subprocess.run(check,capture_output=True).returncode,0)
            self.assertEqual(staged,git("diff","--cached"))
            self.assertEqual((source/"other.txt").read_text(),"user edit\n")
            state=json.loads((root/"upstream.json").read_text());state["revision"]="0"*40;(root/"upstream.json").write_text(json.dumps(state))
            self.assertNotEqual(subprocess.run(check,capture_output=True).returncode,0)
            self.assertEqual((source/"engine.txt").read_text(),"optimized twice\n")
            # A linked worktree and external manifest must never select or edit
            # the default vendor/manifest, including when its index has user edits.
            with tempfile.TemporaryDirectory(prefix="airified-experiment-") as external:
                experiment = Path(external)
                worktree = experiment / "source"
                git("worktree", "add", "--detach", str(worktree), revision)
                manifest = experiment / "upstream.json"
                manifest.write_text(json.dumps({"path":"vendor/strata", "revision":revision, "patches":[]}))
                default_before = (root / "upstream.json").read_bytes()
                canonical_before = git("diff", "HEAD")
                worktree_index = Path(subprocess.check_output(
                    ["git","-C",str(worktree),"rev-parse","--git-path","index"],text=True).strip())
                trial_index_before = worktree_index.read_bytes()
                (worktree / "engine.txt").write_text("isolated experiment\n")
                subprocess.run([sys.executable,str(root/"tools/record_patch.py"),
                                "isolated.patch","engine.txt","--manifest",str(manifest),
                                "--source",str(worktree),"--patch-dir",str(experiment/"patches")],
                               check=True,capture_output=True)
                trial_env = {**os.environ,"UPSTREAM_MANIFEST":str(manifest),"STRATA_SOURCE_DIR":str(worktree)}
                subprocess.run(check,env=trial_env,check=True,capture_output=True)
                self.assertEqual(trial_index_before,worktree_index.read_bytes())
                self.assertEqual(default_before,(root/"upstream.json").read_bytes())
                self.assertEqual(canonical_before,git("diff","HEAD"))
                wrong=json.loads(manifest.read_text());wrong["revision"]="0"*40
                manifest.write_text(json.dumps(wrong))
                self.assertNotEqual(subprocess.run(check,env=trial_env,capture_output=True).returncode,0)
                self.assertEqual((worktree/"engine.txt").read_text(),"isolated experiment\n")



if __name__ == "__main__":
    unittest.main()
