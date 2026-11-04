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


if __name__ == "__main__":
    unittest.main()
