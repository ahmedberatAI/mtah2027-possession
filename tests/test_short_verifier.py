"""Decision rules of the short-possession verifier (src/exp_verify.py apply, used by model_v13)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import exp_verify as ev  # noqa: E402

T_INS, T_DEL = 0.65, 0.35


def row(typ, s, t, k, a, e, min_len=8):
    return dict(typ=typ, s=s, t=t, k=k, a=a, e=e, min_len=min_len)


def apply(pred, rows, p):
    return ev.apply(pred, rows, np.asarray(p, float), True, True, T_INS, T_DEL)


class VerifierApplyTests(unittest.TestCase):
    def test_insert_needs_confidence(self):
        pred = np.zeros(200, np.int8)
        r = row("ins", 80, 110, 1, 0, 200)
        out = apply(pred, [r], [0.9])
        self.assertTrue((out[80:110] == 1).all() and (out[:80] == 0).all() and (out[110:] == 0).all())
        self.assertTrue((apply(pred, [r], [0.6]) == pred).all())

    def test_insert_keeps_decoder_minimum(self):
        pred = np.zeros(200, np.int8)
        self.assertTrue((apply(pred, [row("ins", 4, 40, 1, 0, 200)], [0.9]) == pred).all())
        out = apply(pred, [row("ins", 0, 40, 1, 0, 200)], [0.9])
        self.assertTrue((out[:40] == 1).all() and (out[40:] == 0).all())

    def test_delete_hands_possession_to_the_neighbouring_team(self):
        pred = np.array([0] * 50 + [1] * 40 + [0] * 50, np.int8)
        r = row("del", 50, 90, 1, 50, 90)
        self.assertTrue((apply(pred, [r], [0.2]) == 0).all())
        self.assertTrue((apply(pred, [r], [0.5]) == pred).all())

    def test_delete_next_to_a_stoppage(self):
        pred = np.array([0] * 50 + [1] * 40 + [2] * 50, np.int8)
        out = apply(pred, [row("del", 50, 90, 1, 50, 90)], [0.1])
        self.assertTrue((out[:90] == 0).all() and (out[90:] == 2).all())

    def test_delete_keeps_a_possession_between_stoppages(self):
        pred = np.array([2] * 50 + [1] * 40 + [2] * 50, np.int8)
        self.assertTrue((apply(pred, [row("del", 50, 90, 1, 50, 90)], [0.1]) == pred).all())


if __name__ == "__main__":
    unittest.main()
