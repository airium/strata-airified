import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'vendor/strata/serve/persistent_prefix.py'
spec=importlib.util.spec_from_file_location('persistent_prefix_under_test',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Refused(Exception):
    def __init__(self,kind):self.kind=kind

class Engine:
    def __init__(self):self.last={'reused':0};self.calls=[];self.fail=None;self.tokens=13
    def alive(self):return True
    def session_file(self,action,path):
        self.calls.append((action,path))
        if action=='restore':
            if self.fail:raise Refused(self.fail)
            return {'tokens':self.tokens,'bytes':128,'ms':1}
        Path(path).write_bytes(b'x'*128)
        if self.fail:raise Refused(self.fail)
        return {'tokens':self.tokens,'bytes':128,'ms':1}

class PrefixCache(unittest.TestCase):
    def cache(self,root,key='a'*64,budget=1024):
        obj=module.PersistentPrefixCache(Path(root),key,budget,4,128);obj.bind_layout('0:24,24:48');return obj
    def save(self,cache,engine,ids):
        engine.tokens=len(ids)+1
        cache.before(engine,ids,{},None);cache.after(engine,ids,[91,92],{},threading.Event())
    def test_fresh_object_restore_and_real_hit_accounting(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.cache(root);engine=Engine();ids=list(range(12));self.save(a,engine,ids)
            b=self.cache(root);fresh=Engine();b.before(fresh,ids+[91,92,93],{},None)
            self.assertEqual(fresh.calls[0][0],'restore');self.assertEqual(b.stats['restores'],1)
            fresh.last={'reused':0};b.after(fresh,ids+[91,92,93],[94,95],{},threading.Event())
            self.assertEqual(b.stats['prefix_hits'],0,'file restoration is not a proven prefix hit')
    def test_optout_never_reads_or_writes_files(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.cache(root);engine=Engine();ids=list(range(12));self.save(a,engine,ids)
            before={p.name:p.read_bytes() for p in a.directory.iterdir() if p.is_file()}
            b=self.cache(root);probe=Engine();b.before(probe,ids,{'cache_prompt':False});b.after(probe,ids,[3],{'cache_prompt':False},threading.Event())
            self.assertFalse(probe.calls);self.assertEqual(before,{p.name:p.read_bytes() for p in a.directory.iterdir() if p.is_file()})
    def test_invalid_native_file_falls_back_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.cache(root);ids=list(range(12));self.save(a,Engine(),ids)
            b=self.cache(root);bad=Engine();bad.fail='invalid';b.before(bad,ids,{})
            self.assertEqual(b.stats['errors'],1);self.assertFalse(b.current['restored'])
            calls=len(bad.calls);b.before(bad,ids,{});self.assertEqual(len(bad.calls),calls)
    def test_failed_new_save_preserves_committed_generation(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.cache(root);ids=list(range(12));self.save(a,Engine(),ids)
            meta=a.directory/(module._token_key(ids)+'.json');before=meta.read_bytes();leaf=json.loads(before)['file']
            bad=Engine();bad.fail='io';a.before(bad,ids,{});a.after(bad,ids,[98,99],{},threading.Event())
            self.assertEqual(meta.read_bytes(),before);self.assertTrue((a.directory/leaf).exists())
            self.assertEqual(len(list(a.directory.glob('*.session'))),1)
    def test_budget_never_deletes_foreign_payloads(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.cache(root,budget=128);foreign=a.root/'foreign.session';foreign.write_bytes(b'keep')
            self.save(a,Engine(),list(range(12)));time.sleep(.01);self.save(a,Engine(),list(range(20,32)))
            self.assertEqual(len(list(a.directory.glob('*.session'))),1);self.assertEqual(foreign.read_bytes(),b'keep')
    def test_build_context_and_layer_layout_separate_namespaces(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.cache(root);self.save(a,Engine(),list(range(12)))
            other=self.cache(root,key='b'*64);e=Engine();other.before(e,list(range(12)),{});self.assertFalse(e.calls)
            split=self.cache(root);split.bind_layout('0:25,25:48');e=Engine();split.before(e,list(range(12)),{});self.assertFalse(e.calls)

    def test_namespace_changes_with_build_model_and_precision(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);binary=root/'worker';binary.write_bytes(b'worker1');model=root/'model.gguf';model.write_bytes(b'model1')
            args=['--prompt-cache','6','--max-context','128','--native',str(model),'--kv','int8']
            env={'STRATA_PERSISTENT_CACHE_DIR':str(root/'cache')}
            a,_=module.prepare_environment(binary,args,root,env)
            b,_=module.prepare_environment(binary,args,root,{**env,'BENCH_ADAPTER_PID':'1234'})
            self.assertEqual(a['STRATA_SESSION_BUILD_KEY'],b['STRATA_SESSION_BUILD_KEY'])
            changed=list(args);changed[-1]='fp16';c,_=module.prepare_environment(binary,changed,root,env)
            self.assertNotEqual(a['STRATA_SESSION_BUILD_KEY'],c['STRATA_SESSION_BUILD_KEY'])
            model.write_bytes(b'model2');d,_=module.prepare_environment(binary,args,root,env)
            self.assertNotEqual(a['STRATA_SESSION_BUILD_KEY'],d['STRATA_SESSION_BUILD_KEY'])
            binary.write_bytes(b'worker2');e,_=module.prepare_environment(binary,args,root,env)
            self.assertNotEqual(d['STRATA_SESSION_BUILD_KEY'],e['STRATA_SESSION_BUILD_KEY'])

    def test_pending_cleanup_is_scoped_and_committed_files_survive(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.cache(root);ids=list(range(12));self.save(a,Engine(),ids)
            pending=a.directory/('.'+'f'*64+'.'+'e'*16+'.tmp');pending.write_bytes(b'incomplete')
            foreign=a.directory/'.notes.tmp';foreign.write_bytes(b'keep')
            with a.locked():a._clean_pending()
            self.assertFalse(pending.exists());self.assertEqual(foreign.read_bytes(),b'keep')
            self.assertEqual(len(list(a.directory.glob('*.session'))),1)

    def test_identical_committed_turn_does_not_rewrite_payload(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.cache(root);engine=Engine();ids=list(range(12));self.save(a,engine,ids)
            count=len(engine.calls);a.before(engine,ids,{});a.after(engine,ids,[91,92],{},threading.Event())
            self.assertEqual(len(engine.calls),count)

if __name__=='__main__':unittest.main()
