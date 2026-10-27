"""v12 = v10 + the touch switch relief in the decoder (round 9, src/exp_tswitch.py, src/exp_tcombo.py, src/exp_boot3.py):
ball touches are detected from the tracking (src/touch.py: a sharp change of the ball's velocity with a player within
2 m, credited to that player's team); where the touch sequence changes team, a change of possession INTO that team
within +-6 frames costs 8 less than pen_team (12). No new evidence and no retraining: v10's team probabilities, dead
ball and B2 decoder otherwise. Held out, against v10 (learned / true D): with v10's seed-0 model DFL-trk +0.003 /
+0.003, Metrica-ctrl +0.003 / +0.003, SkillCorner +0.001 / -0.000; averaged over five model instances (v10 seeds 0-1,
v9 seeds 0-2) Metrica +0.0024 / +0.0015, DFL +0.0010 / +0.0003, SkillCorner ~0 -- an option, not the default.

  make_submission.py --model model_v12 [--params '{"wh_bonus": 30}']
"""
import model_v7
import model_v10

DEFAULT = dict(model_v10.DEFAULT, ts_lam=8.0, ts_w=6)


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    return model_v7.predict(f, p, probs_path=model_v10.ensure_probs())
