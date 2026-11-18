"""Portable renderer checks for the opt-in K-quant prompt MMQ route: the native hyper-connection route plus
STRATA_MMQ_KQ (Q4_K gate/up through int8 MMQ) and STRATA_MMQ_DIRECT (resident experts read in place), and explicit
host values still beat the profile."""
import json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NATIVE='q4km-int8kv-256k-dflash2-sm75-tcwide-c50-trim24-fused-hcnative-persistent'
KQMMQ='q4km-int8kv-256k-dflash2-sm75-tcwide-c50-trim24-fused-hcnative-kqmmq-direct-persistent'
FLAGS={'STRATA_MMQ_KQ','STRATA_MMQ_DIRECT'}
class KquantMmqProfile(unittest.TestCase):
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
    def test_only_the_kquant_mmq_flags_differ_from_the_native_route(self):
        a=self.render(NATIVE);b=self.render(KQMMQ)
        self.assertTrue(all(b['env'][x]=='1' for x in FLAGS));self.assertFalse(FLAGS & set(a['env']))
        self.assertEqual({k:v for k,v in b['env'].items() if k not in FLAGS},a['env'])
        self.assertEqual({k:v for k,v in b.items() if k not in ['env','log']},{k:v for k,v in a.items() if k not in ['env','log']})
    def test_explicit_host_value_beats_profile(self):
        cfg=self.render(KQMMQ,STRATA_MMQ_DIRECT='0')
        self.assertEqual(cfg['env']['STRATA_MMQ_DIRECT'],'0');self.assertEqual(cfg['env']['STRATA_MMQ_KQ'],'1')
if __name__=='__main__':unittest.main()
