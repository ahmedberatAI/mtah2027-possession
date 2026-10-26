import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exp_seed_balance import source_weights,crossed_ci


class SeedBalanceTests(unittest.TestCase):
    def test_weights_preserve_total_mass_and_equalize_sources(self):
        source=np.array([0]*3+[1]*29,np.int8);w=source_weights(source)
        self.assertAlmostEqual(float(w[source==0].sum()),float(w[source==1].sum()),places=5)
        self.assertAlmostEqual(float(w.mean()),1.,places=6)
        self.assertTrue((w>0).all())

    def test_weight_ratio_is_inverse_source_size(self):
        w=source_weights(np.array([0]*2+[1]*20))
        self.assertAlmostEqual(float(w[0]/w[-1]),10.,places=6)

    def test_missing_source_is_rejected(self):
        with self.assertRaises(ValueError):source_weights(np.zeros(4,dtype=int))

    def test_constant_paired_difference_has_point_interval(self):
        np.testing.assert_allclose(crossed_ci(np.full((3,20),.04)),[.04,.04])

    def test_shared_match_uncertainty_not_shrunk_by_seed_replication(self):
        a=np.array([[-.3,-.2,0.,.1,.4]])
        np.testing.assert_allclose(crossed_ci(a),crossed_ci(np.repeat(a,3,axis=0)),atol=.02)

    def test_seed_uncertainty_is_kept_with_identical_matches(self):
        lo,hi=crossed_ci(np.repeat(np.array([[-.2],[0.],[.2]]),20,axis=1))
        self.assertLess(lo,-.1);self.assertGreater(hi,.1)


if __name__=='__main__':unittest.main()
