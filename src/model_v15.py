"""v15 = v14 + the short-possession specialist C1 (rounds 23-25, docs/RAPOR.md: C1abETmS).

Team probability: 0.75 x v14's blend (0.5 v10 + 0.5 shared BiGRU equal mean) + 0.25 x C1's live-normalised probability
that the home team controls the ball. C1 (src/short_specialist.py, models/kpu_c1/) is a ball-centred tracking model
trained only on open PFF data, with PFF's labels in the challenge's controlled-possession reading
(gradient.controlled_reading); it never saw DFL, Metrica, SkillCorner or BAR-ATH. Then v10's +-25-frame logit
smoothing, dead ball and B2 decoder, as in v14. The verifier re-judges the short decisions with v13's thresholds on G1 +
C1's five candidate features + its three touch features, with C1's own insert candidates (theta 0.7) next to the
blend's; its classifiers (models/short_verifier_*_c1m25s.txt) are trained on all open-set rows of the evaluated
configuration: v10 seed 0, C1 seed 20261102 (blend, candidates, features), both dead channels, DFL and SkillCorner rows
under the target convention (DFL evctrl, SkillCorner evctrl; Metrica ctrl already is) -- the rows of src/exp_kpu.py
build C1abETmS, seed 0. BAR-ATH uses the same single C1 seed, so the submission is one of the configurations the
round-25 evaluation scored.

C1's BAR-ATH outputs are cached in cache/v15/c1_barath.pkl and the team probabilities in cache/v15/barath_probs.pkl
(deterministic). MTAH_RETRAIN_VERIFIER=1 (or a missing file) retrains the verifier from the research caches of
src/exp_kpu.py.
"""
import os

import numpy as np

import bench
import bfree
import exp_short as es
import exp_short_angles as ea
import exp_smooth
import exp_verify as ev
import model_v7
import model_v9
import model_v10
import model_v13
import short_specialist as ss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_BIGRU = 0.5
W_C1 = 0.25
C1_SEED = ss.SEEDS[0]
C1_MODEL = os.path.join(ROOT, "models", "kpu_c1", "kpu_C1_s%d.pt" % C1_SEED)
C1_OUT = os.path.join(ROOT, "cache", "v15", "c1_barath.pkl")
PROBS = os.path.join(ROOT, "cache", "v15", "barath_probs.pkl")
ROWS = "C1abETmS"                                  # src/exp_kpu.py rows the verifier is trained on
TAG = "c1m25s"
V15 = dict(model=C1_MODEL, out=C1_OUT, probs=PROBS, rows=ROWS, tag=TAG)     # model_v17 runs the same recipe with its own specialist
THETA = 0.7                                        # src/exp_kpu.py THETA["C1"]
KEYS = ev.FEATSETS["G1"] + ["kpu_c", "kpu_cmax", "kpu_gin", "kpu_gout", "kpu_ctx", "kpu_tin", "kpu_tmean", "kpu_tout"]
DEFAULT = dict(model_v10.DEFAULT)
VERIFIER = dict(t_ins=model_v13.VERIFIER["t_ins"], t_del=model_v13.VERIFIER["t_del"])


def ensure_c1(refresh=False, cfg=V15):
    """The specialist's outputs on BAR-ATH's halves, {block start: (T, 9) float16} (stored as the research outputs are)."""
    if os.path.isfile(cfg["out"]) and not refresh:
        return bench._load(cfg["out"])
    model = ss.load(cfg["model"])
    bench._barath_inputs()
    out = {b["a"]: ss.predict(model, ss.features(b["P"], b["B"], b["team"], b["gk"], b["pitch"])).astype(np.float16)
           for b in bench.blocks("barath")}
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    bench._dump(out, cfg["out"])
    return out


def _raws(cfg=V15):
    """{block start: v15's raw team probability} = 0.75 x v14's blend + 0.25 x C1's live-normalised P(home)."""
    gru = bench._load(model_v13.ensure_gru())
    c1 = ensure_c1(cfg=cfg)
    model = model_v10.team_model()
    bench._barath_inputs()
    out = {}
    for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        raw = bench.predict_team(model, dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
        blend = es.blend([raw], [gru[b["a"]][1]], W_BIGRU)[0]
        out[b["a"]] = (1 - W_C1) * blend + W_C1 * ss.pk(c1[b["a"]], 0)
    return out


def ensure_probs(refresh=False, cfg=V15):
    if os.path.isfile(cfg["probs"]) and not refresh:
        return cfg["probs"]
    probs = {a: (None, model_v9.smooth(raw)) for a, raw in _raws(cfg).items()}
    os.makedirs(os.path.dirname(cfg["probs"]), exist_ok=True)
    bench._dump(probs, cfg["probs"])
    return cfg["probs"]


def train_verifier(retrain=None, cfg=V15):
    """The insert / delete classifiers, loaded from models/ unless missing or MTAH_RETRAIN_VERIFIER=1."""
    import lightgbm as lgb
    import short_verifier
    if retrain is None:
        retrain = os.environ.get("MTAH_RETRAIN_VERIFIER") == "1"
    paths = {typ: os.path.join(ROOT, "models", "short_verifier_%s_%s.txt" % (typ, cfg["tag"])) for typ in ("ins", "del")}
    if not retrain and all(os.path.isfile(p) for p in paths.values()):
        return {typ: bench.load_booster(p) for typ, p in paths.items()}
    import exp_kpu
    exp_kpu.use(cfg["rows"])
    rows = [r for src in exp_kpu.SRCS for d in ev.DEADS for r in ev._rows4(src, 0, d)["rows"]]
    exp_kpu.use("v14")
    out = {}
    for typ, path in paths.items():
        sel = [r for r in rows if r["typ"] == typ and r["y"] >= 0]
        X = np.array([[r[k] for k in KEYS] for r in sel], float)
        y = np.array([r["y"] for r in sel])
        out[typ] = lgb.train(short_verifier.PARAMS, lgb.Dataset(X, y), ev.ROUNDS)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out[typ].save_model(path)
    return out


def verify(labels, t_ins, t_del, cfg=V15):
    """The verifier's decisions on BAR-ATH's decoded labels (both halves); returns new labels."""
    models = train_verifier(cfg=cfg)
    gru = bench._load(model_v13.ensure_gru())
    c1 = ensure_c1(cfg=cfg)
    raws = _raws(cfg)
    out = np.asarray(labels).copy()
    min_len = int(model_v7.DEFAULT["min_poss"])
    for b in bench.blocks("barath"):
        raw = raws[b["a"]]
        smo = exp_smooth.smooth([raw], 25)[0]
        o = np.asarray(c1[b["a"]], np.float32)
        sl = slice(b["a"], b["b"] + 1)
        pr = out[sl].astype(np.int8)
        fav = np.where(ea.logit(smo) > 0, 0, 1)
        st = np.flatnonzero(np.r_[True, pr[1:] != pr[:-1]])
        own = [(a + s0, a + t0, 1 - kk) for a, e, kk in zip(st, np.r_[st[1:], len(pr)], pr[st])
               for s0, t0 in ea.runs(fav[a:e] != kk) if kk < 2 and t0 - s0 >= ea.RUN]
        extra = ss.kpu_cands(o, pr, ss.contest_mask(b), THETA, own)
        rows = ev.rows_for("barath", [b], [ea.frame_cues(b)], [pr], [pr], [smo], [raw], [gru[b["a"]][1]], min_len,
                           extra=[extra])
        if not rows:
            continue
        ss.touch_features(ss.add_features(rows, [o]), [o])
        p = np.full(len(rows), np.nan)
        for typ in ("ins", "del"):
            idx = [n for n, r in enumerate(rows) if r["typ"] == typ]
            if idx:
                p[idx] = models[typ].predict(np.array([[rows[n][k] for k in KEYS] for n in idx], float))
        out[sl] = ev.apply(pr, rows, p, True, True, t_ins, t_del)
    return out


def predict(f, params=None, cfg=V15):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    labels = model_v7.predict(f, p, probs_path=ensure_probs(cfg=cfg))
    return verify(labels, **VERIFIER, cfg=cfg)
