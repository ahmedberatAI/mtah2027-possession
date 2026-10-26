import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exp_hybrid import blend
from possession_error_audit import one, diagnostics, complementary


class HybridTests(unittest.TestCase):
    def test_probability_blend_endpoints_and_no_vote(self):
        a=[np.array([.9,.1])];b=[np.array([.3,.8])]
        np.testing.assert_array_equal(blend(a,b,1)[0],a[0])
        np.testing.assert_array_equal(blend(a,b,0)[0],b[0])
        np.testing.assert_allclose(blend(a,b,.25)[0],[.45,.625])
        for x,y,w in ((a,b,1.1),(a,[np.array([.2])],.5),([np.array([np.nan])],[np.array([.5])],.5)):
            with self.assertRaises(ValueError):blend(x,y,w)

    def test_perfect_predictions(self):
        gt=np.repeat([0,1,2,0],[70,30,40,120]).astype(np.int8)
        r,flags=one(gt,gt)
        self.assertTrue(flags.all());self.assertEqual(r['missed'],0)
        self.assertEqual(r['false_positive'],0);self.assertEqual(r['latency_sum_frames'],0)

    def test_short_miss_and_merge_vs_split(self):
        gt=np.repeat([0,1,0],[100,30,100]).astype(np.int8)
        pred=np.zeros(len(gt),np.int8)
        r,_=one(gt,pred)
        self.assertEqual(r['0-2s_gt'],1);self.assertEqual(r['0-2s_missed'],1)
        self.assertEqual(r['merge_proxy'],1)
        reverse,_=one(pred,gt)
        self.assertEqual(reverse['split_proxy'],1)

    def test_late_transition_sign(self):
        gt=np.repeat([0,1],[100,100]).astype(np.int8)
        pred=np.repeat([0,1],[110,90]).astype(np.int8)
        r,_=one(gt,pred)
        self.assertEqual(r['transition_matched'],1)
        self.assertEqual(r['latency_sum_frames'],10)
        self.assertEqual(r['late_transitions'],1)

    def test_noise_draws_are_not_independent_matches(self):
        gt=np.repeat([0,1],[100,100]).astype(np.int8)
        blocks=[dict(group='m',seed=i) for i in range(3)]
        r=diagnostics(blocks,[gt]*3,[gt]*3)
        self.assertEqual(r['counts']['gt'],2)
        self.assertEqual(len(r['counts_by_match']),1)

    def test_complementary_direction_and_unknown_boundaries(self):
        gt=np.repeat([0,3,1],[100,10,100]).astype(np.int8)
        perfect=gt.copy();bad=np.zeros(len(gt),np.int8)
        c=complementary([dict(group='m')],[gt],[bad],[perfect])
        self.assertEqual(c['only_lgb_matches'],1)
        self.assertEqual(c['only_bigru_matches'],0)
        r,_=one(gt,perfect)
        self.assertEqual(r['transition_gt'],0)


if __name__=='__main__':unittest.main()
