"""Round 49: sim.V6 only redraws the ball around dead stretches, is deterministic, and leaves V3 as it was."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import sim  # noqa: E402


def toy(n=30000, absent_dead=False):
    rng = np.random.default_rng(5)
    lab = np.zeros(n, np.int8)
    t = 500
    team = 0
    while t < n - 2000:
        live = int(rng.integers(300, 1500))
        lab[t:t + live] = team
        team = 1 - team
        t += live
        dead = int(rng.integers(60, 800))
        lab[t:t + dead] = 2
        t += dead
    lab[t:] = team
    P = rng.normal(0.5, 0.1, (n, 22, 2)).astype(np.float32)
    B = np.stack([rng.normal(0.5, 0.1, n), rng.normal(0.5, 0.1, n), np.zeros(n), np.zeros(n)], 1)
    if absent_dead:
        B[lab == 2] = np.nan
    return P, B, lab


class EdgeTests(unittest.TestCase):
    def test_live_mask_differs_only_around_restarts(self):
        P, B, lab = toy()
        _, b3 = sim.degrade(P, B, lab, (105.0, 68.0), seed=3, params=sim.V3)
        _, b6 = sim.degrade(P, B, lab, (105.0, 68.0), seed=3, params=sim.V6)
        s3, s6 = np.isfinite(b3[:, 0]), np.isfinite(b6[:, 0])
        restart = np.flatnonzero((lab[1:] != 2) & (lab[:-1] == 2)) + 1
        near = np.zeros(len(lab), bool)
        for r in restart:
            near[r:r + sim.V6["edge_jitter"] + 1] = True
        live = lab != 2
        self.assertTrue(np.array_equal(s3[live & ~near], s6[live & ~near]))
        self.assertFalse(np.array_equal(s3[~live], s6[~live]))

    def test_deterministic(self):
        P, B, lab = toy()
        _, a = sim.degrade(P, B, lab, (105.0, 68.0), seed=7, params=sim.V6)
        _, b = sim.degrade(P, B, lab, (105.0, 68.0), seed=7, params=sim.V6)
        self.assertTrue(np.array_equal(a, b, equal_nan=True))

    def test_players_and_noise_stay_paired_with_v3(self):
        P, B, lab = toy()
        p3, _ = sim.degrade(P, B, lab, (105.0, 68.0), seed=3, params=sim.V3)
        p6, _ = sim.degrade(P, B, lab, (105.0, 68.0), seed=3, params=sim.V6)
        self.assertTrue(np.array_equal(p3, p6, equal_nan=True))

    def test_revealed_frames_without_a_source_hold_a_position(self):
        P, B, lab = toy(absent_dead=True)
        _, b6 = sim.degrade(P, B, lab, (105.0, 68.0), seed=11, params=sim.V6)
        dead = lab == 2
        self.assertGreater(np.isfinite(b6[dead, 0]).mean(), 0.05)
        self.assertTrue(np.all(np.isfinite(b6[np.isfinite(b6[:, 0]), 1])))

    def test_ball_no_longer_returns_on_the_restart_frame(self):
        P, B, lab = toy()
        restart = np.flatnonzero((lab[1:] != 2) & (lab[:-1] == 2)) + 1

        def exact(par):
            _, b = sim.degrade(P, B, lab, (105.0, 68.0), seed=3, params=par)
            s = np.isfinite(b[:, 0])
            return np.mean([s[r] and not s[r - 1] for r in restart])
        self.assertGreater(exact(sim.V3), 0.3)
        self.assertLess(exact(sim.V6), 0.2)


if __name__ == "__main__":
    unittest.main()
