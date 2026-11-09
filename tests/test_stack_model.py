"""The stacker of model_v21 and the ball-free channel: no match data, synthetic input only."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("BF_WIN", "150")
import torch  # noqa: E402

import exp_ballfree as bf  # noqa: E402
import stack_model as sm  # noqa: E402


class StackTests(unittest.TestCase):
    def net(self):
        torch.manual_seed(0)
        net = sm.build_net(sm.N_OWN + 4 + 1 + 1, e=1)
        with torch.no_grad():
            net.t.copy_(torch.tensor([-3.0, -2.0, -6.0, -4.0, -5.0, -5.0, -3.0, -6.0]))
        return net.eval()

    def x(self, T=80, seed=1):
        rng = np.random.default_rng(seed)
        own = rng.normal(0, 1, (T, sm.N_OWN)).astype(np.float32)
        com = rng.normal(0, 1, (T, 4)).astype(np.float32)
        bl = rng.normal(0, 1, (T, 1)).astype(np.float32)
        al = rng.normal(0, 1, (T, 1)).astype(np.float32)
        return own, com, bl, al

    def test_probabilities_and_lengths(self):
        own, com, bl, al = self.x()
        x = np.hstack([own, own[:, ::-1], com, bl, -bl, al])
        q, alive = sm.score_half(self.net(), x, 80 * sm.STEP)
        self.assertEqual(len(q), 80 * sm.STEP)
        self.assertEqual(len(alive), 80 * sm.STEP)
        self.assertTrue(np.all((q >= 0) & (q <= 1)))
        self.assertTrue(np.all(np.isfinite(alive)))

    def test_swapping_the_teams_mirrors_the_answer(self):
        own, com, bl, al = self.x()
        other = np.random.default_rng(2).normal(0, 1, own.shape).astype(np.float32)
        net = self.net()
        a = np.hstack([own, other, com, bl, -bl, al])
        b = np.hstack([other, own, com, -bl, bl, al])
        qa, ala = sm.score_half(net, a, 400)
        qb, alb = sm.score_half(net, b, 400)
        self.assertLess(np.abs(qa - (1 - qb)).max(), 1e-3)
        self.assertLess(np.abs(ala - alb).max(), 1e-3)

    def test_a_move_to_the_other_team_enters_its_control_state(self):
        A = sm.moves(self.net().t)
        self.assertEqual(float(A[0, 3]), sm._FIX)
        self.assertEqual(float(A[1, 3]), sm._FIX)
        self.assertEqual(float(A[2, 1]), sm._FIX)
        self.assertEqual(float(A[3, 1]), sm._FIX)

    def test_load_reads_the_width_off_the_weights(self):
        import tempfile
        d_in = sm.N_OWN + 4 + 1 + 1
        torch.manual_seed(0)
        net = sm.build_net(d_in, width=32, e=1).eval()
        own, com, bl, al = self.x()
        x = np.hstack([own, own[:, ::-1], com, bl, -bl, al])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wide.pt")
            torch.save(dict(state=net.state_dict(), d_in=d_in, e=1), path)
            back = sm.load(path)
        self.assertEqual(back.front[0].weight.shape[0], 32)
        qa, ala = sm.score_half(net, x, 400)
        qb, alb = sm.score_half(back, x, 400)
        self.assertEqual(np.abs(qa - qb).max(), 0.0)
        self.assertEqual(np.abs(ala - alb).max(), 0.0)


class BallFreeTests(unittest.TestCase):
    def blocks(self, n=500):
        rng = np.random.default_rng(0)
        P = np.cumsum(rng.normal(0, 0.0005, (n, 24, 2)), axis=0) + rng.uniform(0.2, 0.8, (1, 24, 2))
        P = P.astype(np.float32)
        P[100:120, 3] = np.nan
        B = rng.uniform(0.2, 0.8, (n, 2)).astype(np.float32)
        team = np.array([0] * 12 + [1] * 12)
        gk = np.zeros(24, bool)
        gk[[0, 12]] = True
        P[:, 0, 0], P[:, 12, 0] = 0.05, 0.95
        return P, B, team, gk

    def test_tokens_are_present_players_only(self):
        P, B, team, gk = self.blocks()
        tok, ball, signs = bf.tokens(P, B, team, gk, (105.0, 68.0))
        self.assertEqual(tok.shape[1:], (2, bf.M, 6))
        self.assertEqual(len(tok), len(ball))
        self.assertEqual(signs.tolist(), [1.0, -1.0])
        self.assertTrue(np.all(tok[:, :, :, 5] <= 1))
        self.assertTrue(np.all(np.abs(tok[tok[..., 5] == 0][:, :5]) == 0))
        self.assertLessEqual(float(tok[:, 0, :, 5].sum(1).max()), 12)

    def test_the_channel_never_reads_the_ball(self):
        P, B, team, gk = self.blocks()
        torch.manual_seed(0)
        net = bf.build_net().eval()
        tok, _, signs = bf.tokens(P, B, team, gk, (105.0, 68.0))
        q1, a1 = bf.predict(net, tok, signs, len(P), "cpu")
        tok2, _, signs2 = bf.tokens(P, B[::-1].copy(), team, gk, (105.0, 68.0))
        q2, a2 = bf.predict(net, tok2, signs2, len(P), "cpu")
        self.assertEqual(np.abs(q1 - q2).max(), 0.0)
        self.assertEqual(np.abs(a1 - a2).max(), 0.0)

    def test_swapping_the_teams_mirrors_the_control_logit(self):
        P, B, team, gk = self.blocks()
        torch.manual_seed(0)
        net = bf.build_net().eval()
        tok, _, signs = bf.tokens(P, B, team, gk, (105.0, 68.0))
        q1, _ = bf.predict(net, tok, signs, len(P), "cpu")
        q2, _ = bf.predict(net, tok[:, ::-1].copy(), signs[::-1].copy(), len(P), "cpu")
        self.assertLess(np.abs(q1 - (1 - q2)).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
