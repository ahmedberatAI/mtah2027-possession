"""Short-possession verifier (round 15, src/exp_verify.py): a second look at v10's short decisions.

Candidates come from the decoded labels and the +-25 team probability: "ins" = runs >= 12 frames inside a decoded
possession where the probability favours the other team (flips the decoder suppressed), "del" = decoded possessions
< 4 s. Two LightGBM classifiers (one per candidate type) score them from identity-free tracking cues, v10's margins and
optionally the shared BiGRU's margin; an "ins" candidate is inserted when p >= t_ins, a "del" possession is handed to
the other team when p < t_del. The classifiers are trained on the open external data only (DFL trk, Metrica ctrl,
SkillCorner phase labels; v10 seed-0 held-out probabilities; learned and true dead ball) and kept in models/, so the
submission can be rebuilt from the provided data. MTAH_RETRAIN_VERIFIER=1 (or a missing file) retrains them; that needs
the research caches of src/exp_verify.py (held-out v10 and BiGRU probabilities on the external sets), so it is not part
of MTAH_RETRAIN=1, which retrains v10's team classifier only and still rebuilds the same submission.
base="blend" with weight w (round 17b: w = .5, src/model_v14.py) uses the decode and margins of the blend
(1 - w) x v10 + w x BiGRU instead of v10's, with its own classifiers (models/short_verifier_*_blend50.txt).
"""
import os

import numpy as np

import bench
import exp_short as es
import exp_short_angles as ea
import exp_smooth
import exp_verify as ev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = dict(ev.PARAMS, deterministic=True, force_row_wise=True, num_threads=8)


def model_path(typ, use_g, base="v10", w=None):
    tag = ""
    if base == "blend":
        assert w is not None, "the blend verifier needs the blend weight"
        tag = "_blend%02d" % round(100 * w)
    return os.path.join(ROOT, "models", "short_verifier_%s%s%s.txt" % (typ, "_g" if use_g else "", tag))


def keys(use_g):
    return ev.BASE + (["m_gru"] if use_g else [])


def train(use_g, retrain=None, base="v10", w=None):
    """The two classifiers, loaded from models/ unless missing or MTAH_RETRAIN_VERIFIER=1. base="blend": candidates
    and margins of the weight-w blend (v10 seed 0 with the BiGRU's equal mean; on DFL the leave-one-match-out BiGRU
    of src/exp_bigru_lomo.py, whose probabilities are in cache/bigru_lomo/)."""
    import lightgbm as lgb
    if retrain is None:
        retrain = os.environ.get("MTAH_RETRAIN_VERIFIER") == "1"
    paths = {typ: model_path(typ, use_g, base, w) for typ in ("ins", "del")}
    if not retrain and all(os.path.isfile(p) for p in paths.values()):
        return {typ: bench.load_booster(p) for typ, p in paths.items()}
    rows = []
    for src, convs, dec, prim in ev.SOURCES:
        blocks = bench.blocks(src)
        gts = bench.labels(src, prim)
        raw = es.v10_raw(0, src)
        gru = ev.gru_mean(src) if use_g else None
        if base == "blend":
            gru = ev.gru_mean4(src)
            raw = es.blend(raw, gru, w)
            gru = gru if use_g else None
        smo = exp_smooth.smooth(raw, 25)
        learned = bench.dead_probs(src, "base") if src == "dfl@v3" else bench.dead_ext("base", "v3", src)
        for pds in (learned, [np.where(g == 2, 1 - es.EPS, es.EPS) for g in gts]):
            dec_all = es.decode(dec, 48.0, pds, smo)
            for i, b in enumerate(blocks):           # one block's frame cues at a time (they carry player positions)
                rows += ev.rows_for(src, [b], [ea.frame_cues(b)], [gts[i]], [dec_all[i]], [smo[i]], [raw[i]],
                                    [gru[i]] if gru is not None else None, int(dec[1]), i0=i)
        bench._blocks.clear()
        bench._labels.clear()
    out = {}
    for typ, path in paths.items():
        sel = [r for r in rows if r["typ"] == typ and r["y"] >= 0]
        X = np.array([[r[k] for k in keys(use_g)] for r in sel], float)
        y = np.array([r["y"] for r in sel])
        out[typ] = lgb.train(PARAMS, lgb.Dataset(X, y), ev.ROUNDS)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out[typ].save_model(path)
    return out


def apply_barath(labels, use_g=True, use_ins=True, use_del=True, t_ins=0.5, t_del=0.5, base="v10", w=None):
    """Verifier decisions on BAR-ATH's decoded labels (both halves); returns new labels. base="blend": the margins are
    the weight-w blend's ((1 - w) v10 + w BiGRU equal mean), as decoded by src/model_v14.py."""
    import bfree
    import model_v7
    import model_v10
    models = train(use_g, base=base, w=w)
    team_model = model_v10.team_model()
    bench._barath_inputs()
    gru_all = None
    if use_g or base == "blend":
        gru_all = bench._load(os.path.join(ROOT, "cache", "shared_bigru", "barath_equal_probabilities.pkl"))
    out = np.asarray(labels).copy()
    blocks = bench.blocks("barath")
    feats = bench.features(model_v7.VERSION, "barath")
    min_len = int(model_v7.DEFAULT["min_poss"])
    for b, f in zip(blocks, feats):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        raw = bench.predict_team(team_model, dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
        if base == "blend":
            raw = es.blend([raw], [gru_all[b["a"]][1]], w)[0]
        smo = exp_smooth.smooth([raw], 25)[0]
        sl = slice(b["a"], b["b"] + 1)
        pr = out[sl].astype(np.int8)
        gru = [gru_all[b["a"]][1]] if use_g else None
        rows = ev.rows_for("barath", [b], [ea.frame_cues(b)], [pr], [pr], [smo], [raw], gru, min_len)
        if not rows:
            continue
        p = np.full(len(rows), np.nan)
        for typ in ("ins", "del"):
            idx = [n for n, r in enumerate(rows) if r["typ"] == typ]
            if idx:
                X = np.array([[rows[n][k] for k in keys(use_g)] for n in idx], float)
                p[idx] = models[typ].predict(X)
        out[sl] = ev.apply(pr, rows, p, use_ins, use_del, t_ins, t_del)
    return out
