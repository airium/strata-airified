"""Portable renderer checks for the native hyper-connection route: it is the fixed 24/24 stage-local route plus
STRATA_HC_NATIVE / STRATA_HC_NATIVE_NO_BF16, and explicit host values still beat the profile."""
import json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TRIM='q4km-int8kv-256k-dflash2-sm75-tcwide-c50-trim24-fused-persistent'
NATIVE='q4km-int8kv-256k-dflash2-sm75-tcwide-c50-trim24-fused-hcnative-persistent'
FLAGS={'STRATA_HC_NATIVE','STRATA_HC_NATIVE_NO_BF16'}
class NativeHcProfile(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'tools').mkdir();(self.root/'source').mkdir()
        for name in ['render_config.py','source_config.py']:shutil.copy2(ROOT/'tools'/name,self.root/'tools'/name)
        shutil.copytree(ROOT/'profiles',self.root/'profiles')
        manifest=self.root/'upstream.json';manifest.write_text(json.dumps({'revision':'fixture','path':str(self.root/'source')}))
        self.env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','GGML_','LLAMA_'))}
        self.env.update(UPSTREAM_MANIFEST=str(manifest),STRATA_SOURCE_DIR=str(self.root/'source'),CPU_NUMA_POLICY='off',
            TARGET_MODEL=str(self.root/'models/first.gguf'),PACK_DIR=str(self.root/'pack'),DRAFT_PACK=str(self.root/'draft.pack'),
            SERVER_BIN=str(self.root/'build/strata'),ENGINE_LIBRARY_PATH=str(self.root/'build'),RUNTIME_DIR=str(self.root/'runtime'),
            PERSISTENT_CACHE_DIR=str(self.root/'kv'),VISION_MODE='cpu',VISION_BIN=str(self.root/'vision'),MMPROJ=str(self.root/'projector.gguf'),
            VISION_THREADS='16',VISION_MAX_TOKENS='1024',GPU_DEVICES='2,3',HOST='127.0.0.1',PORT='3893',CUDA_TOOLKIT=str(self.root/'cuda'))
    def render(self,profile,**extra):
        result=subprocess.run([sys.executable,str(self.root/'tools/render_config.py')],env={**self.env,'PROFILE':profile,**extra},text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr);return json.loads(result.stdout)
    def test_only_the_native_flags_differ_from_the_stage_local_route(self):
        a=self.render(TRIM);b=self.render(NATIVE)
        self.assertTrue(all(b['env'][x]=='1' for x in FLAGS));self.assertFalse(FLAGS & set(a['env']))
        self.assertEqual({k:v for k,v in b['env'].items() if k not in FLAGS},a['env'])
        self.assertEqual({k:v for k,v in b.items() if k not in ['env','log']},{k:v for k,v in a.items() if k not in ['env','log']})
    def test_explicit_host_value_beats_profile(self):
        cfg=self.render(NATIVE,STRATA_HC_NATIVE='0')
        self.assertEqual(cfg['env']['STRATA_HC_NATIVE'],'0');self.assertEqual(cfg['env']['STRATA_HC_NATIVE_NO_BF16'],'1')
if __name__=='__main__':unittest.main()
