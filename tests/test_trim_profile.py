"""Portable renderer integration: preserve host choices across the new route."""
import json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='q4km-int8kv-256k-dflash2-sm75-tcwide-c50-persistent'
TRIM='q4km-int8kv-256k-dflash2-sm75-tcwide-c50-trim24-fused-persistent'
class ProfileRender(unittest.TestCase):
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
 def render(self,profile):
  return subprocess.run([sys.executable,str(self.root/'tools/render_config.py')],env={**self.env,'PROFILE':profile},text=True,capture_output=True)
 def test_host_model_vision_and_precision_are_preserved(self):
  a=self.render(BASE);b=self.render(TRIM);self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(b.returncode,0,b.stderr);a=json.loads(a.stdout);b=json.loads(b.stdout)
  self.assertEqual(b['layer_split'],'24');self.assertEqual(b['gpu'],[2,3]);self.assertEqual(b['vision'],a['vision'])
  self.assertEqual(b['args'].count('--trim-stage-weights'),1);self.assertEqual([x for x in b['args'] if x!='--trim-stage-weights'],a['args'])
  extra={'STRATA_FUSE_CONVAB','STRATA_FUSE_HITS','STRATA_SM75_HEAD_PF'}
  self.assertTrue(all(b['env'][x]=='1' for x in extra));self.assertEqual({k:v for k,v in b['env'].items() if k not in extra},a['env'])
  self.assertEqual({k:v for k,v in b.items() if k not in ['layer_split','args','env','log']},{k:v for k,v in a.items() if k not in ['layer_split','args','env','log']})
 def test_explicit_operator_override_beats_profile(self):
  self.env['STRATA_FUSE_HITS']='0';result=self.render(TRIM);self.assertEqual(result.returncode,0,result.stderr);cfg=json.loads(result.stdout)
  self.assertEqual(cfg['env']['STRATA_FUSE_HITS'],'0');self.assertEqual(cfg['env']['STRATA_FUSE_CONVAB'],'1');self.assertEqual(cfg['env']['STRATA_SM75_HEAD_PF'],'1')
 def test_nonboolean_trimming_does_not_silently_change_loading(self):
  profile=json.loads((self.root/'profiles'/(BASE+'.json')).read_text());profile['trim_stage_weights']='false';(self.root/'profiles/invalid-trim.json').write_text(json.dumps(profile))
  result=self.render('invalid-trim');self.assertNotEqual(result.returncode,0);self.assertIn('trim_stage_weights must be a boolean',result.stderr)
if __name__=='__main__':unittest.main()
