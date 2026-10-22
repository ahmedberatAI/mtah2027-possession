"""v7 = v5's evidence with the round 1-3 CPU ideas (docs/FIKIRLER.md):

  alive/dead  v3 rules + aligned broadcast (unchanged; the learned dead-ball model did not beat them, N1)
  team        LightGBM trained on open DFL degraded with sim.V3 (smooth tracking noise -- V2's per-frame noise
              had doubled every speed feature), native DFL possession labels, features
              base + feat_fix:own_half (NaN-slot fix) + feat_ctrl:control (pressure-aware control, C6)
              + feat_team2:c145 (set-piece formation, team shape, team movement)
  decoder     B2 soft minimum possession length (src/decode2.py): ending a possession spell after d < D0 frames
              costs c * (1 - d / D0) on top of the team-switch penalty; v5's min durations / dead penalty.
  options     wh_bonus > 0: v6's referee-whistle term; clean_ball: drop the ATD's predicted ball frames before the
              team features (src/ballclean.py). Both off by default.

Team probabilities are cached in cache/v7/barath_probs.pkl (rebuilt from the provided data and the open DFL
files by ensure_probs; deterministic: fixed seeds, LightGBM deterministic mode, fixed thread count).
"""
import os

import numpy as np

import atd
import bench
import decode2
import model_v5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "base+feat_fix:own_half+feat_ctrl:control+feat_team2:c145"
SRC = "dfl@v3"
CONV = "native"
PROBS = os.path.join(ROOT, "cache", "v7", "barath_probs.pkl")
PROBS_CLEAN = os.path.join(ROOT, "cache", "v7", "barath_probs_cleanball.pkl")
DEFAULT = dict(model_v5.DEFAULT)
DEFAULT.update(pen_team=12.0, b2_D0=175, b2_c=48.0)


def ensure_probs(refresh=False, clean_ball=False):
    """Train the all-DFL team model and write BAR-ATH per-frame P(home in possession). clean_ball: features from
    the tactical ball with the ATD's predicted (fabricated) frames replaced or dropped (src/ballclean.py)."""
    path = PROBS_CLEAN if clean_ball else PROBS
    if os.path.isfile(path) and not refresh:
        return path
    fd = bench.features(VERSION, SRC)
    # the thread count is pinned: LightGBM's deterministic mode is only reproducible for a fixed num_threads
    model = bench.fit_team(fd, bench.labels(SRC, CONV), params={"num_threads": 8})
    blocks = bench.blocks("barath")
    if clean_ball:
        import ballclean
        from make_submission import merged_features
        feats = [_features(b) for b in ballclean.clean_blocks(blocks, merged_features())]
    else:
        feats = bench.features(VERSION, "barath")
    probs = {b["a"]: (None, bench.predict_team(model, f)) for b, f in zip(blocks, feats)}
    bench._dump(probs, path)
    return path


def _features(b):
    """bench features of one (modified) block, not cached."""
    import mlfeat
    F, Th, Ta = mlfeat.compute(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
    for fn in bench._feat_fns(VERSION):
        f2, th2, ta2 = fn(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
        F.update(f2)
        Th.update(th2)
        Ta.update(ta2)
    fk, tk = sorted(F), sorted(Th)
    return {"XF": np.stack([F[k] for k in fk], 1).astype(np.float32),
            "XH": np.stack([Th[k] for k in tk], 1).astype(np.float32),
            "XA": np.stack([Ta[k] for k in tk], 1).astype(np.float32)}


def spec(p, enter_channel=False):
    """B2 decoder; enter_channel adds a time-varying cost on entering D (channel 0, used for the whistle)."""
    j = np.arange(int(p["b2_D0"]))
    ramp = p["b2_c"] * np.clip(1.0 - (j + 1) / float(p["b2_D0"]), 0, None)
    if not enter_channel:
        return decode2.duration_spec(p["pen_team"], p["pen_dead"], p["min_poss"], p["min_dead"], ramp)
    macros = [dict(emis=0, out=0, L=len(ramp), min_exit=p["min_poss"] - 1, exit_cost=ramp),
              dict(emis=1, out=1, L=len(ramp), min_exit=p["min_poss"] - 1, exit_cost=ramp),
              dict(emis=2, out=2, L=p["min_dead"])]
    pen = [[0, p["pen_team"], p["pen_dead"]], [p["pen_team"], 0, p["pen_dead"]], [p["pen_dead"], p["pen_dead"], 0]]
    return decode2.Spec(macros, pen, [[-1, -1, 0], [-1, -1, 0], [-1, -1, -1]])


def scores(f, p, probs_path=None):
    model_v5._probs.clear()
    model_v5.team_logp(f, probs_path or ensure_probs())
    try:
        return model_v5.scores(f, p)
    finally:
        model_v5._probs.clear()


def predict(f, params=None, probs_path=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    S = scores(f, p, probs_path or ensure_probs(clean_ball=bool(p.get("clean_ball", False))))
    sp = spec(p)
    labels = np.full(f["n"], 3, np.int8)
    for _, a, b in atd.HALVES:
        labels[a:b + 1] = sp.decode(S[a:b + 1])
    if p.get("wh_bonus", 0) > 0:
        # v6's referee-whistle evidence: a whistle followed (within wh_lead frames) by a later alive -> D switch
        # rewards entering D at the whistle (+-wh_e0/e1 frames); the labels above decide which whistles are "late"
        import model_v6
        enter = model_v6.enter_bonus(f["n"], dict(model_v6.DEFAULT, **{k: v for k, v in p.items()
                                                                       if k.startswith("wh_")}), labels)
        spe = spec(p, enter_channel=True)
        for _, a, b in atd.HALVES:
            labels[a:b + 1] = spe.decode(S[a:b + 1], -enter[a:b + 1, 2:3])
    return labels
