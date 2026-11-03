import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import dead_timing as dtm  # noqa: E402

H, A, D = 0, 1, 2


def seq(*parts):
    return np.concatenate([np.full(n, k, np.int8) for k, n in parts])


class RetimeTest(unittest.TestCase):
    def test_bounds(self):
        f, k = dtm.dbounds(seq((H, 5), (D, 5), (A, 5)))
        np.testing.assert_array_equal(f, [5, 10])
        np.testing.assert_array_equal(k, [dtm.STOP, dtm.RESTART])

    def test_stop_and_restart_move_onto_the_reference(self):
        lab = seq((H, 50), (D, 40), (A, 50))
        ref = seq((A, 60), (D, 40), (H, 40))            # the reference's teams do not matter
        out = dtm.retime(lab, ref)
        np.testing.assert_array_equal(out, seq((H, 60), (D, 40), (A, 40)))

    def test_restarts_only(self):
        lab = seq((H, 50), (D, 40), (A, 50))
        ref = seq((H, 60), (D, 20), (A, 60))
        out = dtm.retime(lab, ref, kinds=(dtm.RESTART,))
        np.testing.assert_array_equal(out, seq((H, 50), (D, 30), (A, 60)))

    def test_outside_the_window_or_across_a_boundary_stays(self):
        lab = seq((H, 100), (D, 100), (A, 100))
        far = seq((H, 160), (D, 100), (A, 40))           # stop 60 frames later: beyond +-50
        np.testing.assert_array_equal(dtm.retime(lab, far), lab)
        lab2 = seq((H, 40), (A, 10), (D, 100), (A, 50))
        across = seq((H, 35), (D, 115), (A, 50))         # would cross the H -> A change at 40
        np.testing.assert_array_equal(dtm.retime(lab2, across), lab2)

    def test_in_play_changes_are_untouched(self):
        lab = seq((H, 30), (A, 30), (H, 30))
        np.testing.assert_array_equal(dtm.retime(lab, seq((H, 90))), lab)


if __name__ == "__main__":
    unittest.main()
