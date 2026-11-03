import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import soccertrack as stk  # noqa: E402

H, A, D, N = stk.H, stk.A, stk.D, stk.NEUTRAL


class NativeTest(unittest.TestCase):
    def test_neutral_keeps_the_team_that_had_the_ball(self):
        out = stk.native(np.array([H, H, N, N, A, A, N, H], np.int8))
        np.testing.assert_array_equal(out, [H, H, H, H, A, A, A, H])

    def test_neutral_after_dead_ball_takes_the_next_team(self):
        out = stk.native(np.array([D, D, N, N, A, A], np.int8))
        np.testing.assert_array_equal(out, [D, D, A, A, A, A])

    def test_neutral_does_not_cross_dead_ball(self):
        out = stk.native(np.array([H, N, D, N, N, D], np.int8))
        np.testing.assert_array_equal(out, [H, H, D, D, D, D])


if __name__ == "__main__":
    unittest.main()
