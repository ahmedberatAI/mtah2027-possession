"""The Bayes decision on the stacker's joint posterior (src/stack_decode.py): synthetic input only, no match data."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import stack_decode as sd  # noqa: E402

STATE_OF = {0: 0, 1: 2, 2: 4}      # label -> a state of that label in the five-state path model


def certain(labels, strength=30.0):
    """Emissions that leave no doubt about the label of every 5 Hz step."""
    e = np.full((len(labels), 5), -strength)
    for t, k in enumerate(labels):
        e[t, STATE_OF[k]] = 0.0
    A = np.full((5, 5), -3.0)
    np.fill_diagonal(A, 0.0)
    return e, A


def steps(*parts):
    return np.concatenate([np.full(n, k, np.int8) for k, n in parts])


class DecideTests(unittest.TestCase):
    def test_samples_follow_a_certain_posterior(self):
        lab = steps((0, 40), (1, 10), (2, 30), (1, 50))
        e, A = certain(lab)
        seg, off = sd.ffbs(e, A, 20, 7)
        self.assertEqual(len(off), 21)
        for j in range(20):
            y = np.zeros(len(lab), np.int8)
            for a, b, k in seg[off[j]:off[j + 1]]:
                y[a:b] = k
            self.assertTrue(np.array_equal(y, lab))

    def test_a_certain_posterior_is_followed_whatever_the_reference(self):
        lab = steps((0, 40), (1, 10), (2, 30), (1, 50))
        e, A = certain(lab)
        ref = np.zeros(len(lab) * sd.STEP, np.int8)          # a reference that is wrong almost everywhere
        out = sd.decide([(e, A)], ref, 0.5, 50, 3)
        self.assertTrue(np.array_equal(out, np.repeat(lab, sd.STEP)))

    def test_deterministic_for_a_seed(self):
        rng = np.random.default_rng(0)
        e = rng.normal(0, 1, (200, 5))
        A = np.full((5, 5), -2.0)
        np.fill_diagonal(A, 0.0)
        ref = np.repeat(rng.integers(0, 3, 200), sd.STEP).astype(np.int8)
        a = sd.decide([(e, A)], ref, 0.5, 100, 11)
        b = sd.decide([(e, A)], ref, 0.5, 100, 11)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(len(a), len(ref))
        self.assertTrue(set(np.unique(a)) <= {0, 1, 2})

    def test_more_latent_states_per_team(self):
        lab = steps((1, 30), (0, 30))
        e = np.full((60, 7), -30.0)                         # three states per team, then dead
        e[:30, 3] = 0.0
        e[30:, 0] = 0.0
        A = np.full((7, 7), -3.0)
        np.fill_diagonal(A, 0.0)
        seg, off = sd.ffbs(e, A, 5, 1)
        y = np.zeros(60, np.int8)
        for a, b, k in seg[off[0]:off[1]]:
            y[a:b] = k
        self.assertTrue(np.array_equal(y, lab))


if __name__ == "__main__":
    unittest.main()
