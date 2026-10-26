import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exp_barath_public import pack_probs


class BarathPublicTests(unittest.TestCase):
    def test_preserves_absolute_half_frame_offsets(self):
        p=pack_probs([{'a':10,'b':12},{'a':20,'b':21}],
                     [np.array([.1,.2,.3]),np.array([.4,.5])])
        self.assertEqual(set(p),{10,20});np.testing.assert_equal(p[20][1],[.4,.5])

    def test_rejects_off_by_one_and_bad_probabilities(self):
        for value in ([.1],[.1,.2,.3],[.1,np.nan],[.1,1.1]):
            with self.assertRaises(ValueError):pack_probs([{'a':10,'b':11}],[np.array(value)])

    def test_rejects_duplicate_and_missing_halves(self):
        with self.assertRaises(ValueError):pack_probs([{'a':10,'b':10}],[])
        with self.assertRaises(ValueError):pack_probs([{'a':10,'b':10}]*2,[np.array([.5])]*2)


if __name__=='__main__':unittest.main()
