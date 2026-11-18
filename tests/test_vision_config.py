import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class VisionConfig(unittest.TestCase):
    def render(self, _check=False, **settings):
        env = {key: value for key, value in os.environ.items() if not key.startswith("VISION_")}
        with tempfile.TemporaryDirectory() as runtime:
            env.update(PROFILE="q4km-int8kv-256k-dflash2-sm75-tcwide-c50-persistent",
                       GPU_DEVICES="2,3", RUNTIME_DIR=runtime, **settings)
            return subprocess.run([sys.executable, str(ROOT / "tools/render_config.py"), *(["--check"] if _check else [])],
                                  env=env, text=True, capture_output=True)

    def test_disabled_is_the_existing_text_route(self):
        result = self.render(VISION_MODE="off")
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = json.loads(result.stdout)
        self.assertNotIn("vision", cfg)
        self.assertNotIn("--vision", cfg["args"])
        self.assertNotIn("--vram-reserve-mib", cfg["args"])

    def test_cpu_enables_native_vision_without_taking_expert_vram(self):
        cfg = json.loads(self.render(VISION_MODE="cpu", MMPROJ="/models/matched-f16.gguf").stdout)
        self.assertIn("--vision", cfg["args"])
        self.assertFalse(cfg["vision"]["gpu"])
        self.assertEqual(cfg["vision"]["max_tokens"], 1024)
        self.assertEqual(cfg["vision"]["mmproj"], "/models/matched-f16.gguf")
        self.assertNotIn("--vram-reserve-mib", cfg["args"])
        self.assertEqual(cfg["gpu"], [2, 3])

    def test_auto_image_budget_follows_the_reader_device(self):
        for mode, value, expected in [("cpu", None, 1024), ("gpu", None, 4096), ("cpu", "auto", 1024),
                                      ("gpu", "AUTO", 4096), ("gpu", "2048", 2048), ("cpu", "4096", 4096)]:
            with self.subTest(mode=mode, value=value):
                settings = {"VISION_MODE": mode} if value is None else {"VISION_MODE": mode, "VISION_MAX_TOKENS": value}
                result = self.render(**settings)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["vision"]["max_tokens"], expected)

    def test_gpu_keeps_physical_reader_assignment_separate_from_text_pair(self):
        cfg = json.loads(self.render(VISION_MODE="gpu", VISION_CUDA_DEVICE="3").stdout)
        self.assertEqual(cfg["gpu"], [2, 3])
        self.assertEqual(cfg["vision"]["cuda_device"], 3)
        self.assertFalse(cfg["vision"]["elastic"], "layer-split caches cannot lend VRAM dynamically")
        self.assertEqual(cfg["vision"]["cpu_exe"], cfg["vision"]["exe"])
        self.assertEqual(cfg["args"][cfg["args"].index("--vram-reserve-mib") + 1], "700")

    def test_rejects_unbounded_or_invalid_vision_limits(self):
        for settings, name in [({"VISION_MODE":"unknown"}, "VISION_MODE"),
                               ({"VISION_MODE":"gpu", "VISION_MAX_TOKENS":"0"}, "VISION_MAX_TOKENS"),
                               ({"VISION_MODE":"cpu", "VISION_MAX_TOKENS":"big"}, "VISION_MAX_TOKENS"),
                               ({"VISION_MODE":"cpu", "VISION_THREADS":"bad"}, "VISION_THREADS"),
                               ({"VISION_MODE":"gpu", "VISION_CUDA_DEVICE":"-1"}, "VISION_CUDA_DEVICE")]:
            with self.subTest(settings=settings):
                result = self.render(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)

    def test_launch_preflight_checks_each_library_directory_not_the_joined_search_path(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in ['worker','reader','target.gguf','projector.gguf','draft.pack']:
                (folder/name).touch()
            (folder/'pack/tokenizer').mkdir(parents=True)
            (folder/'engine-libs').mkdir();(folder/'cuda-libs').mkdir()
            settings=dict(VISION_MODE='gpu',SERVER_BIN=str(folder/'worker'),VISION_BIN=str(folder/'reader'),
                          TARGET_MODEL=str(folder/'target.gguf'),MMPROJ=str(folder/'projector.gguf'),
                          DRAFT_PACK=str(folder/'draft.pack'),PACK_DIR=str(folder/'pack'),
                          ENGINE_LIBRARY_PATH=str(folder/'engine-libs'),VISION_LIBRARY_PATH=str(folder/'cuda-libs'))
            result=self.render(_check=True,**settings)
            self.assertEqual(result.returncode,0,result.stderr)
            settings['VISION_LIBRARY_PATH']=str(folder/'absent-cuda-libs')
            result=self.render(_check=True,**settings)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('absent-cuda-libs',result.stderr)
