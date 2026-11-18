"""The HTTP frontend closes at EOS instead of exhausting the native generator."""
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/strata'))
try:
    from serve.server import StrataEngine
except ModuleNotFoundError as error:
    raise unittest.SkipTest('run with the Strata requirements environment: '+str(error))


class StreamBoundary(unittest.TestCase):
    def engine(self,finish='stop',generated=1):
        events=[]
        cache=SimpleNamespace(live=[9],before=lambda *args:events.append('lookup'),
                              after=lambda *args:events.append('save'))
        engine=SimpleNamespace(last={},persistent_cache=cache,persistent_lock=threading.RLock())
        def stream(*args):
            try:yield 248046
            finally:
                engine.last={'finish':finish,'generated':generated}
                events.append('native-DONE')
        engine._generate_tokens=stream
        return engine,events

    def test_eos_close_saves_only_after_native_done(self):
        engine,events=self.engine();gen=StrataEngine.generate(engine,[1,2],16,{},threading.Event())
        self.assertEqual(next(gen),248046);gen.close()
        self.assertEqual(events,['lookup','native-DONE','save'])

    def test_unobserved_native_output_is_not_committed_on_close(self):
        engine,events=self.engine(generated=2);gen=StrataEngine.generate(engine,[1,2],16,{},threading.Event())
        next(gen);gen.close()
        self.assertEqual(events,['lookup','native-DONE']);self.assertEqual(engine.persistent_cache.live,[])

    def test_unsettled_error_is_not_committed(self):
        engine,events=self.engine(finish='error');gen=StrataEngine.generate(engine,[1,2],16,{},threading.Event())
        next(gen);gen.close()
        self.assertNotIn('save',events)
