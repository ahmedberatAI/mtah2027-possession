"""Re-derive models/dead_rules_open_v1.json, the constants of model_v19's rule terms, from open data only (rounds 43 and 44).

Thresholds (speed look-ahead, ball margin, unseen time, stillness, fast ball) by provider-balanced log-loss on the seven DFL matches, the three Metrica
games with their ball de-glued and the twenty SkillCorner matches; weights by logistic regression of true alive on the terms; the two view weights on
SkillCorner alone, whose flag is real. BAR-ATH plays no part. Needs the open data (src/download_idsse.py, src/metrica_sample.py, src/skillcorner.py).

    python -I src/dead_rules_fit.py          prints the result and how far it lies from the shipped file
    python -I src/dead_rules_fit.py write    also rewrites the file
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import model_v19  # noqa: E402
import skillcorner as scr  # noqa: E402

SOURCES = {"dfl": ("dfl@v3", "dfl@v3", "evctrl"), "metrica": ("metrica@v3g", "metrica@v3", "ctrl"), "sc": ("sc@v3", "sc@v3", "evctrl")}
GRID = dict(shift=(0, 12, 25, 37, 50, 75), out_margin=(0.0, 0.5, 1.0, 2.0), missing_after=(12, 25, 50), still_disp=(0.3, 0.6, 1.0), ball_fast=(3.0, 5.0, 7.0))
START = dict(shift=25, out_margin=1.0, missing_after=25, still_disp=1.0, ball_fast=5.0)
STRIDE = 10


def view_cues():
    out = []
    for m in scr.matches():
        d = scr.parse(m)
        for h in (1, 2):
            out.append((d["h%d_valid" % h], d["h%d_ndet" % h]))
    return out


def load():
    """provider -> list of (rule fields, truth, valid, live)."""
    data = {}
    for k, (fsrc, lsrc, conv) in SOURCES.items():
        blocks = bench.blocks(fsrc)
        labs = bench.labels(lsrc, conv)
        cues = view_cues() if k == "sc" else [(None, None)] * len(blocks)
        data[k] = []
        for b, g, (v, n) in zip(blocks, labs, cues):
            data[k].append((model_v19.rule_fields(b), np.asarray(g), v, (v & (n >= model_v19.MIN_VISIBLE)) if v is not None else None))
        bench._blocks.clear()
        bench._labels.clear()
    return data


def bounds(data):
    """Tempo clip bounds: the pooled 2nd and 98th percentiles."""
    v1 = np.concatenate([f["spd_all_s25"][::STRIDE] for k in data for f, *_ in data[k]])
    v3 = np.concatenate([f["spd_all_s100"][::STRIDE] for k in data for f, *_ in data[k]])
    return dict(v1=np.nanquantile(v1, (0.02, 0.5, 0.98)).tolist(), v3=np.nanquantile(v3, (0.02, 0.5, 0.98)).tolist())


def logistic(X, y, w, offset=None, it=25, ridge=1e-3):
    """Weighted logistic regression by Newton steps; returns the coefficients and the weighted log-loss."""
    beta = np.zeros(X.shape[1])
    off = 0.0 if offset is None else offset
    w = w / w.mean()
    for _ in range(it):
        z = np.clip(X @ beta + off, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (w * (y - p)) - ridge * len(y) * beta
        h = (X * (w * np.maximum(p * (1 - p), 1e-6))[:, None]).T @ X + ridge * len(y) * np.eye(X.shape[1])
        d = np.linalg.solve(h, g)
        beta += d
        if np.abs(d).max() < 1e-5:
            break
    z = np.clip(X @ beta + off, -30, 30)
    p = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
    return beta, float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / len(y))


def design(data, conf):
    """(X, y, weights) over all providers, each the same total weight."""
    Xs, ys, ws = [], [], []
    for k in data:
        xk, yk = [], []
        for f, g, _, _ in data[k]:
            m = g[::STRIDE] < 3
            xk.append(model_v19.rule_terms(f, conf)[::STRIDE][m])
            yk.append((g[::STRIDE][m] < 2).astype(float))
        x, y = np.concatenate(xk), np.concatenate(yk)
        Xs.append(x)
        ys.append(y)
        ws.append(np.full(len(y), 1.0 / len(y)))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def fit():
    data = load()
    conf = dict(bounds=bounds(data), thresholds=dict(START))
    for name, vals in GRID.items():
        best = None
        for v in vals:
            trial = dict(conf, thresholds=dict(conf["thresholds"], **{name: v}))
            X, y, w = design(data, trial)
            _, ll = logistic(X, y, w)
            if best is None or ll < best[0] - 1e-5:
                best = (ll, v)
        conf["thresholds"][name] = best[1]
    X, y, w = design(data, conf)
    beta, _ = logistic(X, y, w)
    Xv, yv, off = [], [], []
    for f, g, v, l in data["sc"]:
        m = g[::STRIDE] < 3
        Xv.append(np.stack([(~v)[::STRIDE][m], l[::STRIDE][m]], 1).astype(np.float64))
        yv.append((g[::STRIDE][m] < 2).astype(float))
        off.append((model_v19.rule_terms(f, conf) @ beta)[::STRIDE][m])
    gamma, _ = logistic(np.concatenate(Xv), np.concatenate(yv), np.ones(sum(len(a) for a in yv)), offset=np.concatenate(off))
    conf["beta"], conf["gamma"] = beta.tolist(), gamma.tolist()
    return conf


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    new = fit()
    with open(model_v19.RULES) as fh:
        old = json.load(fh)
    print("thresholds", new["thresholds"], "same as shipped:", new["thresholds"] == old["thresholds"])
    print("largest difference from the shipped weights: %.6f" % max(np.abs(np.array(new["beta"] + new["gamma"]) - np.array(old["beta"] + old["gamma"]))))
    if len(sys.argv) > 1 and sys.argv[1] == "write":
        new["note"] = old.get("note", "")
        with open(model_v19.RULES, "w", newline="\n") as fh:
            json.dump(new, fh, indent=1)
        print("written")
