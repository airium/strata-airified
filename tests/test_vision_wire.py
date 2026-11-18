"""Image identity must fit the native uint64 wire field, never saturate/alias."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/strata'))
try:
    from serve.server import StrataEngine
except ModuleNotFoundError as error:
    raise unittest.SkipTest('run with the Strata requirements environment: '+str(error))


class ImageWireKeys(unittest.TestCase):
    def test_production_64_bit_keys_are_preserved(self):
        keys=['a8aa3b0fcab0b860','55cbc9e5ca692095']
        wire=StrataEngine.sampling_keys({'strata_img_keys':keys})
        self.assertIn('img_keys='+','.join(keys),wire)

    def test_overflowing_or_invalid_keys_fall_back_to_native_image_hashes(self):
        for keys in [['a'*32],['f'*17],[''],['g1'],['a8aa3b0fcab0b860','f'*32]]:
            with self.subTest(keys=keys):
                wire=StrataEngine.sampling_keys({'strata_img_keys':keys})
                self.assertNotIn('img_keys=',wire)
