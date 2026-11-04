"""v19 = v17's team side with dead-ball evidence free of the public window (rounds 43 and 44, docs/RAPOR.md: A25).

The official rules say the five public minutes are not a tuning set, and the rules of model_v2 / model_v3 were written looking at them. Here
nothing about the dead ball comes from BAR-ATH. The decoder's alive log-odds per frame are

    0.25 x open-data rules + 0.75 x a multi-source dead model + the view terms

  open-data rules   model_v2's form (player tempo over +-1 s and +-3 s looked ahead, ball out, unseen, still, fast) with every constant from the
                    open sets: thresholds by provider-balanced log-loss, weights by logistic regression (models/dead_rules_open_v1.json,
                    src/exp_rules_open.py fit)
  multi-source      a LightGBM on three providers with equal total weight: the seven DFL matches with their ball status, the three Metrica games
                    with their ball de-glued and their own D, the twenty SkillCorner matches with D from their dynamic events; simulated
                    degradation, tracking features only (models/dead_multi_v1.txt)
  view terms        broadcast without a valid homography, broadcast valid with at least four players visible; both weights from SkillCorner's
                    real view flag

The decoder's settings and the team probability are v17's; the verifier follows as in v17. Frames before the second kick-off (official FAQ) are dead.

Cached: cache/v19/barath_dead.pkl (deterministic). MTAH_RETRAIN_DEAD=1 (or a missing file) retrains the multi-source model from the open data.
"""
import json
import os

import numpy as np

import atd
import bench
import dead_timing
import features as FT
import model_v7
import model_v15
import model_v17

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "dead_multi_v1.txt")
RULES = os.path.join(ROOT, "models", "dead_rules_open_v1.json")
DEAD = os.path.join(ROOT, "cache", "v19", "barath_dead.pkl")
VERSION = "base"
DEFAULT = dict(model_v15.DEFAULT)
W_RULES, W_MODEL = 0.25, 0.75
MIN_VISIBLE = 4
# (features source, labels source, labels convention)
SOURCES = {"dfl": ("dfl@v3", "dfl", "native"), "metrica": ("metrica@v3g", "metrica@v3", "ctrl"), "sc": ("sc@v3", "sc@v3", "evctrl")}


def fit_dead(parts):
    """A dead model on several providers, each provider the same total weight: parts = {provider: (features, labels)}."""
    import lightgbm as lgb
    Xs, ys, ws = [], [], []
    for fs, ls in parts.values():
        x = np.concatenate([f["XF"][::bench.STRIDE] for f in fs])
        y = np.concatenate([np.asarray(l)[::bench.STRIDE] for l in ls])
        g = y < 3
        Xs.append(x[g])
        ys.append((y[g] == 2).astype(int))
        ws.append(np.full(int(g.sum()), 1.0 / int(g.sum())))
    w = np.concatenate(ws)
    params = dict(bench.LGB_DEAD, deterministic=True, force_row_wise=True, num_threads=8)
    return lgb.train(params, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys), weight=w * len(w) / w.sum()), bench.ROUNDS)


def dead_model(retrain=None):
    if retrain is None:
        retrain = os.environ.get("MTAH_RETRAIN_DEAD") == "1"
    if os.path.isfile(MODEL) and not retrain:
        return bench.load_booster(MODEL)
    model = fit_dead({k: (bench.features(VERSION, v[0]), bench.labels(v[1], v[2])) for k, v in SOURCES.items()})
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    model.save_model(MODEL)
    return model


def ensure_dead(refresh=False):
    """{block start: the multi-source model's P(dead) per frame} on BAR-ATH's halves."""
    if os.path.isfile(DEAD) and not refresh:
        return bench._load(DEAD)
    model = dead_model()
    bench._barath_inputs()
    out = {b["a"]: model.predict(f["XF"]) for b, f in zip(bench.blocks("barath"), bench.features(VERSION, "barath"))}
    os.makedirs(os.path.dirname(DEAD), exist_ok=True)
    bench._dump(out, DEAD)
    return out


def rule_fields(b):
    """The tracking fields the rule terms read, from a bench block."""
    L, W = b["pitch"]
    pxy = b["P"][:, :, :2] * np.array([L, W], np.float32)
    bxy = b["B"][:, :2] * np.array([L, W], np.float32)
    seen = np.isfinite(bxy[:, 0])
    present = np.isfinite(pxy[:, :, 0])
    bvel = FT.ball_kinematics(bxy, seen)
    out_x = np.maximum(-bxy[:, 0], bxy[:, 0] - L)
    out_y = np.maximum(-bxy[:, 1], bxy[:, 1] - W)
    spd = np.where(present, np.linalg.norm(FT.player_velocity(pxy), axis=2), np.nan)
    with np.errstate(invalid="ignore"):
        s = np.nanmean(spd, axis=1)
    return {"n": len(bxy), "seen": seen, "bxy": bxy, "bspeed": np.linalg.norm(bvel, axis=1), "ball_out": np.where(seen, np.maximum(out_x, out_y), np.nan),
            "since_seen": FT.gap_context(seen)[0], "spd_all_s25": FT.rolling_mean(s, 25), "spd_all_s100": FT.rolling_mean(s, 100)}


def rule_terms(f, conf):
    """Per frame: intercept, tempo over +-1 s and +-3 s (clipped, looked ahead), ball out, ball unseen, ball still, ball fast."""
    t, b = conf["thresholds"], conf["bounds"]
    n = f["n"]
    s = int(t["shift"])
    v1 = np.clip(np.nan_to_num(np.roll(f["spd_all_s25"], -s), nan=b["v1"][1]), b["v1"][0], b["v1"][2])
    v3 = np.clip(np.nan_to_num(np.roll(f["spd_all_s100"], -s), nan=b["v3"][1]), b["v3"][0], b["v3"][2])
    out = np.nan_to_num(f["ball_out"], nan=-1) > t["out_margin"]
    seen = f["seen"]
    miss = (~seen) & (f["since_seen"] > t["missing_after"])
    lo, hi = np.clip(np.arange(n) - 12, 0, n - 1), np.clip(np.arange(n) + 12, 0, n - 1)
    ok = seen & seen[lo] & seen[hi]
    disp = np.where(ok, np.linalg.norm(f["bxy"][hi] - f["bxy"][lo], axis=1), np.nan)
    still = ok & (disp < t["still_disp"])
    inside = np.nan_to_num(f["ball_out"], nan=1) <= 0
    fast = seen & inside & (np.nan_to_num(f["bspeed"]) > t["ball_fast"])
    return np.stack([np.ones(n), v1, v3, out, miss, still, fast], 1).astype(np.float64)


def alive_evidence(f):
    """The decoder's alive log-odds per frame of the halves."""
    with open(RULES) as fh:
        conf = json.load(fh)
    beta, gamma = np.asarray(conf["beta"]), np.asarray(conf["gamma"])
    dead = ensure_dead()
    valid, nvis = np.asarray(f["b_valid"]), np.asarray(f["b_nvis"])
    a = np.zeros(f["n"])
    for b in bench.blocks("barath"):
        sl = slice(b["a"], b["b"] + 1)
        rules = rule_terms(rule_fields(b), conf) @ beta
        view = gamma[0] * (~valid[sl]) + gamma[1] * (valid[sl] & (nvis[sl] >= MIN_VISIBLE))
        a[sl] = W_RULES * rules + W_MODEL * dead_timing.alive_logit(dead[b["a"]]) + view
    return a


def scores(f, probs_path):
    a = alive_evidence(f)
    lh = np.full(f["n"], np.log(0.5))
    la = np.full(f["n"], np.log(0.5))
    for s, (_, ph) in bench._load(probs_path).items():
        ph = np.clip(ph, 1e-4, 1 - 1e-4)
        lh[s:s + len(ph)] = np.log(ph)
        la[s:s + len(ph)] = np.log(1 - ph)
    S = np.stack([a / 2 + lh, a / 2 + la, -a / 2], 1)
    S[atd.HALVES[1][1]:atd.KICKOFF_2ND, 2] += 1e6
    return S


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    cfg = model_v17.CONFIG
    S = scores(f, model_v15.ensure_probs(cfg=cfg))
    sp = model_v7.spec(p)
    labels = np.full(f["n"], 3, np.int8)
    for _, a, e in atd.HALVES:
        labels[a:e + 1] = sp.decode(S[a:e + 1])
    return model_v15.verify(labels, **model_v15.VERIFIER, cfg=cfg)
