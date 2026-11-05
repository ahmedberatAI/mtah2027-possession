"""Round 51: the sequence dead model's per-half normalisation and the shipped files of v20."""
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["TD_CPU"] = "1"
import exp_tdead as td  # noqa: E402
import model_v19  # noqa: E402
import model_v20  # noqa: E402
import torch  # noqa: E402


class HalfNormTests(unittest.TestCase):
    def test_level_of_a_half_is_removed(self):
        rng = np.random.default_rng(0)
        x = rng.normal(2.0, 0.5, (5000, 3))
        a = td.half_norm(x)
        b = td.half_norm(x * 1.4 + 3.0)
        self.assertTrue(np.allclose(a, b, atol=1e-9))
        self.assertLess(np.abs(np.median(a, axis=0)).max(), 1e-9)

    def test_constant_and_missing_columns(self):
        x = np.ones((100, 2))
        x[:50, 1] = np.nan
        out = td.half_norm(x)
        self.assertTrue(np.all(out[:, 0] == 0))
        self.assertTrue(np.all(np.isnan(out[:50, 1])))

    def test_pool_applies_it_only_when_asked(self):
        x = np.arange(100, dtype=float).reshape(50, 2)
        os.environ.pop("TD_NORM", None)
        plain = td.pool(x)
        os.environ["TD_NORM"] = "1"
        norm = td.pool(x)
        os.environ.pop("TD_NORM")
        self.assertEqual(plain.shape, (10, 2))
        self.assertFalse(np.allclose(plain, norm))


class ShippedTests(unittest.TestCase):
    def test_rules_keep_v1_thresholds(self):
        with open(model_v19.RULES) as fh:
            v1 = json.load(fh)
        with open(model_v20.RULES) as fh:
            v2 = json.load(fh)
        self.assertEqual(v1["thresholds"], v2["thresholds"])
        self.assertEqual(v1["bounds"], v2["bounds"])
        self.assertEqual((len(v2["beta"]), len(v2["gamma"])), (7, 2))

    def test_checkpoints_load_and_score(self):
        rng = np.random.default_rng(1)
        for seed in model_v20.TD_SEEDS:
            ck = torch.load(os.path.join(model_v20.TD_DIR, "tdead_v6n_s%d.pt" % seed), map_location="cpu", weights_only=False)
            net = td.Net(ck["n_in"]).to(td.DEVICE)
            net.load_state_dict(ck["state"])
            net.eval()
            x = rng.normal(size=(1000, ck["n_in"]))
            os.environ["TD_NORM"] = "1"
            p = td.score_half(net, ck["mean"], ck["scale"], x)
            os.environ.pop("TD_NORM")
            self.assertEqual(len(p), 1000)
            self.assertTrue(np.all((p >= 0) & (p <= 1)))

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(model_v20.W_RULES + model_v20.W_M + model_v20.W_TD, 1.0)


if __name__ == "__main__":
    unittest.main()
