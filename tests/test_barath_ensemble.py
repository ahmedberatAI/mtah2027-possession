import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exp_barath_ensemble import equal_probability_mean


class EnsembleTests(unittest.TestCase):
    def test_equal_soft_mean_preserves_frame_alignment(self):
        halves=[(1,10,11),(2,20,20)]
        members=[{10:(None,np.array(x)),20:(None,np.array([.5]))}
                 for x in ([.9,.1],[.4,.4],[.4,.7])]
        out=equal_probability_mean(members,halves)
        np.testing.assert_allclose(out[10][1],[1.7/3,1.2/3])
        self.assertGreater(out[10][1][0],.5)  # soft mean, not majority of hard labels
        self.assertEqual(set(out),{10,20})

    def test_rejects_missing_seed_or_different_half(self):
        member={10:(None,np.array([.5]))}
        with self.assertRaises(ValueError):equal_probability_mean([member]*2,[(1,10,10)])
        with self.assertRaises(ValueError):equal_probability_mean([member,member,{20:(None,np.array([.5]))}],[(1,10,10)])

    def test_rejects_bad_length_nan_and_changed_dead_channel(self):
        for value in ((None,np.array([.2,.3])),(None,np.array([np.nan])),
                      (None,np.array([1.1])),(np.array([.5]),np.array([.5]))):
            with self.assertRaises(ValueError):equal_probability_mean([{10:value}]*3,[(1,10,10)])


if __name__=='__main__':unittest.main()
