import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServeVisionArgument(unittest.TestCase):
    def run_serve(self, *args, config="VISION_MODE=cpu\n"):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.env"
            path.write_text(config)
            env = {key: value for key, value in os.environ.items() if not key.startswith(("STRATA_", "VISION_"))}
            env.update(CONFIG_FILE=str(path), PYTHON_BIN=sys.executable, RUNTIME_DIR=str(Path(temp) / "runtime"),
                       PROFILE="q4km-int8kv-256k-dflash2-sm75-tcwide-c50-trim24-fused-hcnative-persistent",
                       GPU_DEVICES="2,3")
            return subprocess.run(["bash", str(ROOT / "serve.sh"), *args], env=env, capture_output=True, text=True)

    def vision(self, *args, **kw):
        done = self.run_serve(*args, "--print-config", **kw)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout).get("vision")

    def test_config_mode_is_used_without_the_argument(self):
        self.assertEqual((self.vision()["gpu"], self.vision()["max_tokens"]), (False, 1024))

    def test_argument_overrides_the_config_and_follows_the_auto_budget(self):
        for args in (("--vision", "gpu"), ("--vision=gpu",)):
            with self.subTest(args=args):
                cfg = self.vision(*args)
                self.assertTrue(cfg["gpu"])
                self.assertEqual(cfg["max_tokens"], 4096)
                self.assertEqual(cfg["cuda_device"], 2)
        self.assertFalse(self.vision("--vision", "cpu", config="VISION_MODE=gpu\n")["gpu"])
        self.assertIsNone(self.vision("--vision", "off"))

    def test_rejects_a_bad_or_missing_mode(self):
        for args in (("--vision", "tpu"), ("--vision",)):
            with self.subTest(args=args):
                done = self.run_serve(*args, "--print-config")
                self.assertEqual(done.returncode, 2)
                self.assertIn("--vision" if len(args) == 1 else "VISION_MODE", done.stderr)


if __name__ == "__main__":
    unittest.main()
