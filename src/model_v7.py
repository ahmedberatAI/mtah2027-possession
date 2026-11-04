"""v7 = v5's evidence with the round 1-3 CPU ideas (docs/RAPOR.md):

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


def spec(p, enter_channel=False, switch_channels=False):
    """B2 decoder; enter_channel adds a time-varying cost on entering D (channel 0, used for the whistle);
    switch_channels adds time-varying costs on changes of team (channel 1: into A, 2: into H; touch option)."""
    j = np.arange(int(p["b2_D0"]))
    ramp = p["b2_c"] * np.clip(1.0 - (j + 1) / float(p["b2_D0"]), 0, None)
    if not enter_channel and not switch_channels:
        return decode2.duration_spec(p["pen_team"], p["pen_dead"], p["min_poss"], p["min_dead"], ramp)
    macros = [dict(emis=0, out=0, L=len(ramp), min_exit=p["min_poss"] - 1, exit_cost=ramp),
              dict(emis=1, out=1, L=len(ramp), min_exit=p["min_poss"] - 1, exit_cost=ramp),
              dict(emis=2, out=2, L=p["min_dead"])]
    pen = [[0, p["pen_team"], p["pen_dead"]], [p["pen_team"], 0, p["pen_dead"]], [p["pen_dead"], p["pen_dead"], 0]]
    e = 0 if enter_channel else -1
    sa, sh = (1, 2) if switch_channels else (-1, -1)
    return decode2.Spec(macros, pen, [[-1, sa, e], [sh, -1, e], [-1, -1, -1]])


def touch_terms(n, p):
    """Touch option (round 9, src/exp_tswitch.py / src/exp_tanchor.py): per frame of the match, the anchor signal
    clip(sum of +-1 over touches within [-12, +3] frames, -1, 1) (+1 home) and the switch-relief indicators (into A,
    into H) within +-ts_w frames of a touch that changes the touch team."""
    import touch
    anc = np.zeros(n)
    sw = np.zeros((n, 2))
    for b in bench.blocks("barath"):
        L, W = b["pitch"]
        m = len(b["P"])
        t, k = touch.touches(b["P"][:, :, :2] * np.array([L, W], np.float32),
                             b["B"][:, :2] * np.array([L, W], np.float32), np.asarray(b["team"]))
        acc = np.zeros(m + 1)
        s = np.where(k == 0, 1.0, -1.0)
        np.add.at(acc, np.clip(t - 12, 0, m), s)
        np.add.at(acc, np.clip(t + 4, 0, m), -s)
        anc[b["a"]:b["a"] + m] = np.clip(np.cumsum(acc[:-1]), -1, 1)
        if len(t) > 1:
            g = np.flatnonzero(k[1:] != k[:-1]) + 1
            for col, team in ((0, 1), (1, 0)):
                acc = np.zeros(m + 1)
                sel = t[g[k[g] == team]]
                np.add.at(acc, np.clip(sel - p.get("ts_w", 6), 0, m), 1)
                np.add.at(acc, np.clip(sel + p.get("ts_w", 6) + 1, 0, m), -1)
                sw[b["a"]:b["a"] + m, col] = np.cumsum(acc[:-1]) > 0
    return anc, sw


def anchored_probs(probs_path, beta, anc):
    """A copy of a team-probability file with the touch anchors fused into the logit (beta * anchor signal)."""
    out = probs_path.replace(".pkl", "_touchanchor%g.pkl" % beta)
    probs = bench._load(probs_path)
    fused = {}
    for a, (pd_, ph) in probs.items():
        z = np.log(np.clip(ph, 1e-4, 1 - 1e-4) / np.clip(1 - ph, 1e-4, 1 - 1e-4)) + beta * anc[a:a + len(ph)]
        fused[a] = (pd_, 1 / (1 + np.exp(-z)))
    bench._dump(fused, out)
    return out


def scores(f, p, probs_path=None):
    model_v5._probs.clear()
    model_v5.team_logp(f, probs_path or ensure_probs())
    try:
        return model_v5.scores(f, p)
    finally:
        model_v5._probs.clear()


def po_fuse(S, p):
    """Players-only dead-ball evidence (src/podead.py; round 9): the dead channel's log-odds move towards the
    players-only model's by lam * clip(l_po - l_base, -c, c) -- bounded, lam = 0 leaves v7 unchanged."""
    import podead_barath
    po = podead_barath.probs()[:len(S)]
    lb = S[:, 2] - np.logaddexp(S[:, 0], S[:, 1])
    lp = np.log(np.clip(po, 1e-4, 1 - 1e-4) / np.clip(1 - po, 1e-4, 1 - 1e-4))
    d = p["po_lam"] * np.clip(lp - lb, -p["po_c"], p["po_c"])
    if p.get("po_gate", "all") == "uncertain":       # only where v7's own dead-ball evidence is weak (0.1 < p < 0.9)
        d = np.where(np.abs(lb) < np.log(9.0), d, 0.0)
    S = S.copy()
    S[:, 2] += np.where(np.isfinite(d), d, 0.0)
    return S


def predict(f, params=None, probs_path=None, alive=None):
    """alive: optional extra alive log-odds per frame, added to the dead-ball evidence (model_v18's learned model)."""
    p = dict(DEFAULT)
    if params:
        p.update(params)
    probs_path = probs_path or ensure_probs(clean_ball=bool(p.get("clean_ball", False)))
    ts = p.get("ts_lam", 0) > 0 or p.get("ts_beta", 0) > 0     # touch option, off by default
    if ts:
        anc, sw = touch_terms(f["n"], p)
        if p.get("ts_beta", 0) > 0:
            probs_path = anchored_probs(probs_path, p["ts_beta"], anc)
        relief = -p.get("ts_lam", 0.0) * sw
    S = scores(f, p, probs_path)
    if p.get("po_lam", 0) > 0:
        S = po_fuse(S, p)
    if alive is not None:
        S = S + np.stack([alive / 2, alive / 2, -alive / 2], axis=1)
    sp = spec(p, switch_channels=ts)
    labels = np.full(f["n"], 3, np.int8)
    for _, a, b in atd.HALVES:
        if ts:
            labels[a:b + 1] = sp.decode(S[a:b + 1], np.hstack([np.zeros((b + 1 - a, 1)), relief[a:b + 1]]))
        else:
            labels[a:b + 1] = sp.decode(S[a:b + 1])
    if p.get("wh_bonus", 0) > 0:
        # v6's referee-whistle evidence: a whistle followed (within wh_lead frames) by a later alive -> D switch
        # rewards entering D at the whistle (+-wh_e0/e1 frames); the labels above decide which whistles are "late"
        import model_v6
        enter = model_v6.enter_bonus(f["n"], dict(model_v6.DEFAULT, **{k: v for k, v in p.items()
                                                                       if k.startswith("wh_")}), labels)
        spe = spec(p, enter_channel=True, switch_channels=ts)
        for _, a, b in atd.HALVES:
            chan = -enter[a:b + 1, 2:3]
            if ts:
                chan = np.hstack([chan, relief[a:b + 1]])
            labels[a:b + 1] = spe.decode(S[a:b + 1], chan)
    return labels
