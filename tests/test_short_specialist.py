"""The challenge's controlled-possession reading of the label sources (gradient.controlled_reading, skillcorner.pp_events;
rounds 24 / 25) and the decision-time pieces of the short-possession specialist (src/short_specialist.py, used by
src/model_v15.py and src/exp_kpu.py)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import gradient  # noqa: E402
import short_specialist as ss  # noqa: E402
import skillcorner as scr  # noqa: E402

H, A, D = 0, 1, 2


def outputs(n, ph):
    """(n, 9) specialist outputs with P(home) = ph among live frames and no events."""
    o = np.zeros((n, 9), np.float32)
    o[:, 0], o[:, 1] = ph, 1 - ph
    return o


class ControlledReadingTests(unittest.TestCase):
    LAB = np.array([H] * 100 + [A] * 30 + [H] * 100, np.int8)
    BACK = [(135, H, "start", None), (140, H, "PA", "C")]     # the home team regains with a completed pass

    def read(self, evs, lab=None):
        return gradient.controlled_reading(self.LAB if lab is None else lab, evs + self.BACK)

    def test_regaining_needs_control_too(self):
        self.assertTrue((gradient.controlled_reading(self.LAB, [(100, A, "start", None), (105, A, "PA", "C")])[100:]
                         == A).all())

    def test_touch_without_control_goes_back(self):
        self.assertTrue((self.read([(100, A, "start", None)]) == H).all())

    def test_completed_pass_keeps_the_spell(self):
        self.assertTrue((self.read([(100, A, "start", None), (105, A, "PA", "C")]) == self.LAB).all())

    def test_failed_pass_right_away_does_not(self):
        self.assertTrue((self.read([(100, A, "start", None), (105, A, "PA", None)]) == H).all())

    def test_event_after_half_a_second_keeps_the_spell(self):
        self.assertTrue((self.read([(100, A, "start", None), (113, A, "possession_loss", None)]) == self.LAB).all())
        self.assertTrue((self.read([(100, A, "start", None), (105, A, "possession_loss", None)]) == H).all())

    def test_spell_after_a_stoppage_is_kept(self):
        lab = np.array([H] * 100 + [D] * 50 + [A] * 30 + [H] * 100, np.int8)
        self.assertTrue((gradient.controlled_reading(lab, [(185, H, "PA", "C")]) == lab).all())


class SkillCornerEventsTests(unittest.TestCase):
    def test_player_possessions_in_the_reading_vocabulary(self):
        pp = pd.DataFrame(dict(team_id=[7, 9, 7], frame_start=[1010, 1020, 1040], frame_end=[1014, 1030, 1044],
                               end_type=["pass", "shot", "pass"], pass_outcome=["successful", np.nan, "unsuccessful"]))
        self.assertEqual(scr.pp_events(pp, 7, 1000, 105),        # 10 fps -> 25 fps grid; frames past n are dropped
                         [(25, H, "start", None), (35, H, "PA", "C"), (50, A, "start", None), (75, A, "SH", None),
                          (100, H, "start", None)])


class SpecialistUseTests(unittest.TestCase):
    def test_runs(self):
        m = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 1], bool)
        self.assertEqual(ss.runs(m, 2), [(1, 4), (5, 7)])
        self.assertEqual(ss.runs(m, 1), [(1, 4), (5, 7), (9, 10)])

    def test_control_is_live_normalised(self):
        o = np.zeros((1, 9), np.float32)
        o[0, :3] = 0.2, 0.2, 0.6
        self.assertAlmostEqual(float(ss.pk(o, 0)[0]), 0.5)

    def test_insert_candidates(self):
        n = 200
        pr = np.zeros(n, np.int8)                              # a home possession throughout
        ph = np.full(n, 0.9, np.float32)
        ph[80:90] = 0.2                                        # away in control for 10 frames
        o = outputs(n, ph)
        contest = np.zeros(n, bool)
        contest[85] = True
        self.assertEqual(ss.kpu_cands(o, pr, contest, 0.7, []), [(80, 90, A)])
        self.assertEqual(ss.kpu_cands(o, pr, np.zeros(n, bool), 0.7, []), [])            # never contested
        self.assertEqual(ss.kpu_cands(o, pr, contest, 0.7, [(78, 92, A)]), [])           # the blend already has it
        self.assertEqual(ss.kpu_cands(o, pr, contest, 0.7, [(78, 92, H)]), [(80, 90, A)])
        self.assertEqual(ss.kpu_cands(o, pr, contest, 0.85, []), [])                     # below theta

    def test_candidate_features(self):
        n = 200
        ph = np.full(n, 0.9, np.float32)
        ph[80:90] = 0.2
        o = outputs(n, ph)
        o[78, 4], o[95, 6], o[83, 8], o[94, 7] = 0.7, 0.6, 0.9, 0.5   # away gains, away loses, away / home touches
        r = [dict(i=0, s=80, t=90, k=A)]
        ss.touch_features(ss.add_features(r, [o]), [o])
        want = dict(kpu_c=0.8, kpu_cmax=0.8, kpu_gin=0.7, kpu_gout=0.6, kpu_ctx=0.1, kpu_tin=0.9, kpu_tmean=0.09,
                    kpu_tout=0.5)
        for k, v in want.items():
            self.assertAlmostEqual(r[0][k], v, places=5, msg=k)

    def test_players_per_team(self):
        """Round 34: k players per team (C3: 11); the default layout is C1's, and the width tells k back."""
        rng = np.random.default_rng(1)
        team = np.array([0] * 11 + [1] * 11)
        gk = np.zeros(22, bool)
        gk[[0, 11]] = True
        P, B = rng.uniform(size=(60, 22, 2)).astype(np.float32), rng.uniform(size=(60, 2)).astype(np.float32)
        x5, x11 = ss.features(P, B, team, gk, (105.0, 68.0)), ss.features(P, B, team, gk, (105.0, 68.0), k=11)
        self.assertEqual((x5.shape[-1], x11.shape[-1]), (ss.N_BASE, ss.n_base(11)))
        self.assertEqual((ss._width_k(ss.N_BASE), ss._width_k(ss.N_BASE + 2), ss._width_k(ss.n_base(11))), (5, 5, 11))
        self.assertEqual(ss.y_cols(ss.K), ss.Y_COLS)
        self.assertTrue(np.array_equal(x5[:, :, :8 + 7 * 5], x11[:, :, :8 + 7 * 5]))       # ball + own nearest five
        self.assertTrue(np.array_equal(x5[:, :, -12:], x11[:, :, -12:]))                   # context block


class V15Tests(unittest.TestCase):
    def test_verifier_files_match_the_keys(self):
        import bench
        import model_v15
        for typ in ("ins", "del"):
            m = bench.load_booster(os.path.join(ROOT, "models", "short_verifier_%s_%s.txt" % (typ, model_v15.TAG)))
            self.assertEqual(m.num_feature(), len(model_v15.KEYS))

    def test_shipped_specialist_runs(self):
        import model_v15
        rng = np.random.default_rng(0)
        n = 160
        team = np.array([0] * 11 + [1] * 11)
        gk = np.zeros(22, bool)
        gk[[0, 11]] = True
        x = ss.features(rng.uniform(size=(n, 22, 2)).astype(np.float32), rng.uniform(size=(n, 2)).astype(np.float32),
                        team, gk, (105.0, 68.0))
        out = ss.predict(ss.load(model_v15.C1_MODEL), x)
        self.assertEqual(out.shape, (n, 9))
        self.assertTrue(np.allclose(out[:, :3].sum(1), 1, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
