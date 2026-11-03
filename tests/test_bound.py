import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import exp_bound as eb  # noqa: E402

H, A, D = 0, 1, 2


def seq(*parts):
    return np.concatenate([np.full(n, k, np.int8) for k, n in parts])


class BoundTest(unittest.TestCase):
    def test_changes_skip_dead_ball(self):
        f, k = eb.changes(seq((H, 30), (A, 30), (D, 30), (H, 30)))
        np.testing.assert_array_equal(f, [30])
        np.testing.assert_array_equal(k, [A])

    def test_twins_same_direction_nearest_one_to_one(self):
        pred = seq((H, 100), (A, 100), (H, 100))
        gt = seq((H, 110), (A, 60), (H, 130))
        np.testing.assert_array_equal(eb.twins(pred, gt), [10, -30])

    def test_twin_missing_beyond_window_or_other_direction(self):
        pred = seq((H, 100), (A, 100))
        np.testing.assert_array_equal(np.isnan(eb.twins(pred, seq((H, 160), (A, 40)))), [True])
        np.testing.assert_array_equal(np.isnan(eb.twins(pred, seq((A, 100), (H, 100)))), [True])

    def test_shift_moves_and_respects_neighbours(self):
        pred = seq((H, 100), (A, 100), (H, 100))
        rows = [{"f": 100}, {"f": 200}]
        out = eb.shift(pred, rows, [10.0, -95.0], [0.9, 0.9])
        # first change later by 10; the second may not leave A shorter than 8 frames after the first moved
        np.testing.assert_array_equal(out, seq((H, 110), (A, 8), (H, 182)))

    def test_shift_needs_confidence_and_a_real_move(self):
        pred = seq((H, 100), (A, 100))
        np.testing.assert_array_equal(eb.shift(pred, [{"f": 100}], [20.0], [0.4]), pred)
        np.testing.assert_array_equal(eb.shift(pred, [{"f": 100}], [1.0], [0.9]), pred)

    def test_shift_never_reverses_direction(self):
        pred = seq((H, 5), (A, 100))                      # spell before the change already shorter than 8 frames
        np.testing.assert_array_equal(eb.shift(pred, [{"f": 5}], [-3.0], [0.9]), pred)


if __name__ == "__main__":
    unittest.main()
