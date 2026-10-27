import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exp_shared_blend import blend, bootstrap, by_match


class SharedBlendTests(unittest.TestCase):
    def test_logit_blend_retains_equal_opposing_evidence(self):
        actual = blend([np.array([.1, .8])], [np.array([.9, .2])], "logit", .5)
        np.testing.assert_allclose(actual[0], [.5, .5])

    def test_endpoint_probabilities_are_preserved_before_smoothing(self):
        a, b = [np.array([0., 1.])], [np.array([.6, .2])]
        for method in ("probability", "logit"):
            np.testing.assert_array_equal(blend(a, b, method, 0)[0], a[0])
            np.testing.assert_array_equal(blend(a, b, method, 1)[0], b[0])
        with self.assertRaises(ValueError):
            blend(a, [np.array([.5])], "logit", .5)

    def test_match_bootstrap_does_not_count_noise_draws_as_games(self):
        result = bootstrap({"a": .8, "b": .5, "c": .7}, {"a": .7, "b": .4, "c": .6})
        self.assertEqual(result["matches"], 3)
        np.testing.assert_allclose(result["ci95"], [.1, .1])

    def test_noise_seeds_are_averaged_before_match_aggregation(self):
        from unittest.mock import patch
        groups = {("game1", 0): (.9,), ("game1", 1): (.7,), ("game2", 0): (.5,)}
        with patch("exp_shared_blend.bench.spell_by_group", return_value=groups):
            self.assertEqual(by_match([], [], []), {"game1": .8, "game2": .5})


if __name__ == "__main__":
    unittest.main()
