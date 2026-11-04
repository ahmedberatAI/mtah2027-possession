"""Round 40: the view flag transplant keeps a segment's state, and the fused evidence is the plain sum."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import dead_timing as dt  # noqa: E402
import exp_deadmix as dm  # noqa: E402


class TransplantTests(unittest.TestCase):
    def setUp(self):
        dm._lib.clear()
        dm._lib.update({False: [(40, np.ones(40, bool), np.ones(40, bool)), (400, np.ones(400, bool), np.zeros(400, bool))],
                        True: [(60, np.zeros(60, bool), np.zeros(60, bool))]})

    def tearDown(self):
        dm._lib.clear()

    def test_segments_take_their_own_state(self):
        g = np.array([0] * 100 + [2] * 77 + [1] * 13 + [2] * 200 + [0] * 5)
        valid, live = dm.transplant(g, np.random.default_rng(0))
        self.assertEqual(len(valid), len(g))
        self.assertTrue(np.array_equal(valid, g != 2))
        self.assertFalse(live[g == 2].any())

    def test_stretching_covers_every_frame(self):
        dm._lib[True] = [(4, np.array([True, False, True, False]), np.zeros(4, bool))]
        g = np.full(10, 2)
        valid, _ = dm.transplant(g, np.random.default_rng(0))
        self.assertEqual(valid.tolist(), [True, True, True, False, False, True, True, True, False, False])


class EvidenceTests(unittest.TestCase):
    def test_mix_is_rules_plus_weighted_learned(self):
        a, b = [np.array([1.0, -2.0])], [np.array([4.0, 6.0])]
        self.assertTrue(np.allclose(dm.mix_evidence(a, b)[0], [3.0, 1.0]))
        self.assertTrue(np.allclose(dm.mix_evidence(a, b, 0.25)[0], [2.0, -0.5]))

    def test_alive_logit(self):
        self.assertAlmostEqual(float(dt.alive_logit(np.array([0.5]))[0]), 0.0)
        x = dt.alive_logit(np.array([0.0, 1.0]))
        self.assertAlmostEqual(float(x[0]), -float(x[1]))
        self.assertGreater(float(x[0]), 9.0)


class OpenRulesTests(unittest.TestCase):
    """model_v19 scores the rule terms the fit in src/exp_rules_open.py used."""

    def setUp(self):
        import json
        import model_v19
        self.mv = model_v19
        with open(model_v19.RULES) as fh:
            self.conf = json.load(fh)
        rng = np.random.default_rng(0)
        n = 600
        P = rng.uniform(size=(n, 22, 2)).astype(np.float32)
        B = np.cumsum(rng.normal(scale=0.002, size=(n, 2)), axis=0).astype(np.float32) + 0.5
        B[100:160] = np.nan
        self.b = dict(P=P, B=B, pitch=(105.0, 68.0))

    def test_same_terms_as_the_fit(self):
        import exp_blend
        import exp_rules_open as ro
        ours = self.mv.rule_fields(self.b)
        theirs = exp_blend.rule_f(self.b)
        np.testing.assert_array_equal(self.mv.rule_terms(ours, self.conf), ro.terms(theirs, self.conf["thresholds"], self.conf["bounds"]))

    def test_weights_have_the_expected_signs(self):
        beta, gamma = self.conf["beta"], self.conf["gamma"]
        self.assertEqual(len(beta), 7)
        self.assertGreater(beta[2], 0)      # tempo means alive
        self.assertLess(beta[3], 0)         # ball out means dead
        self.assertLess(gamma[0], 0)        # no valid view means dead
        self.assertAlmostEqual(self.mv.W_RULES + self.mv.W_MODEL, 1.0)


if __name__ == "__main__":
    unittest.main()
