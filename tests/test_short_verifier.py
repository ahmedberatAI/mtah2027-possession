"""Decision rules of the short-possession verifier (src/exp_verify.py apply, used by model_v13 and model_v14)."""
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


class BlendVerifierTests(unittest.TestCase):
    """model_v14's blend weight reaches both the probabilities and the verifier's model files (round 17b)."""

    def test_blend_weight(self):
        import exp_short as es
        a, b = [np.array([0.2, 0.6])], [np.array([0.8, 0.0])]
        self.assertTrue(np.allclose(es.blend(a, b, 0.5)[0], [0.5, 0.3]))
        self.assertTrue(np.allclose(es.blend(a, b)[0], (1 - es.W_BIGRU) * a[0] + es.W_BIGRU * b[0]))

    def test_model_files(self):
        import model_v14
        import short_verifier as sv
        self.assertEqual(os.path.basename(sv.model_path("ins", True)), "short_verifier_ins_g.txt")      # v13's
        self.assertEqual(os.path.basename(sv.model_path("del", True, "blend", model_v14.W_BIGRU)),
                         "short_verifier_del_g_blend50.txt")
        self.assertRaises(AssertionError, sv.model_path, "ins", True, "blend")
        self.assertEqual(model_v14.VERIFIER["w"], model_v14.W_BIGRU)


if __name__ == "__main__":
    unittest.main()
