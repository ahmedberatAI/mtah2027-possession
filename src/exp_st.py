"""Round 28 (docs/RAPOR.md): SoccerTrack v2 (src/soccertrack.py; ten university matches, CC BY 4.0) -- what it holds and
how the shipped configurations do on it. No component of v14 / v15 saw these matches: v10's team classifier (DFL +
Metrica), the shared BiGRU (DFL + PFF), C1 (PFF) and the verifiers (DFL, Metrica, SkillCorner rows). Each run uses the
open-data components of a shipped configuration: v10's full-data classifier of seed k (cache/bench/combo/
a2_bfree_all*_wm2.txt), the BiGRU's three members (equal mean), C1 of seed k (v15), +-25 smoothing, the B2 decoder
(30, 8, 50), learned (base, v3) or true dead ball, and the shipped verifiers with their thresholds. Inputs are degraded
with sim.V3 like every external set. Labels: the provider's ball status in the target convention ('evctrl', the
controlled reading of src/gradient.py, keep 0.5 s) and as is ('native'). The ball is rebuilt from the events (anchored
at them), so absolute scores are optimistic; the comparison is paired by match. AGGREGATE output only.

  python -I src/exp_st.py measure     labels, provider status vs events, ball anchoring, tracking, pixel scale
  python -I src/exp_st.py eval        v14 against v15 on the ten matches (three seeds)
"""
import json
import os
import re
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_smooth  # noqa: E402
import exp_verify as ev  # noqa: E402
import metrica_sample as ms  # noqa: E402
import model_v7  # noqa: E402
import short_specialist as ss  # noqa: E402
import soccertrack as stk  # noqa: E402

SRC = "st@v3"
OUT = os.path.join(bench.CACHE, "st")
DEC = (30.0, 8, 50)
CONVS = ("evctrl", "native")
NAMES = ("v14", "v15")


# -- measure ------------------------------------------------------------------------------------------

def _measure_one(m):
    d = stk.parse(m)
    evs = stk.events(m, d)
    res = {"labels": {c: stk.labels(m, c) for c in CONVS}, "status": [], "agree": [], "anchor": [], "kink": [],
           "npl": [], "moves": []}
    L, W = stk.PITCH
    slot = {int(p): j for j, p in enumerate(d["pids"])}
    for h in (1, 2):
        res["status"].append(np.bincount(d["h%d_status" % h], minlength=4))
        nat = res["labels"]["native"][h - 1]
        e = evs[h - 1]
        res["agree"] += [int(nat[f] == k) for f, k, *_ in e if nat[f] < 2]
        B = d["h%d_ball" % h].astype(float) * (L, W)
        P = d["h%d_players" % h].astype(float) * (L, W)
        res["anchor"] += [float(np.hypot(*(B[f] - P[f, slot[pid]]))) for f, k, kind, oc, pid in e
                          if np.isfinite(B[f]).all() and np.isfinite(P[f, slot[pid]]).all()]
        w = 5                                       # direction change over 0.2 s windows (positions on a ~1 m grid)
        d1, d2 = B[w:-w] - B[:-2 * w], B[2 * w:] - B[w:-w]
        n1, n2 = np.hypot(*d1.T), np.hypot(*d2.T)
        ok = (n1 >= 1) & (n2 >= 1)
        kink, moving = np.zeros(len(B), bool), np.zeros(len(B), bool)
        kink[w:-w] = ok & (np.sum(d1 * d2, 1) / np.maximum(n1 * n2, 1e-9) < np.cos(np.radians(30)))
        moving[w:-w] = ok
        evf = np.array(sorted({f for f, *_ in e}))
        near = np.zeros(len(B), bool)
        for o in range(-2, 3):
            near[np.clip(evf + o, 0, len(B) - 1)] = True
        res["kink"].append((kink[evf].sum(), moving[evf].sum(), kink[~near].sum(), moving[~near].sum(),
                            (kink & near).sum(), kink.sum()))
        alive = nat < 2
        res["npl"].append(np.bincount(np.isfinite(P[alive, :, 0]).sum(1), minlength=40)[:40])
        dP = np.abs(np.diff(P, axis=0)).sum(2)
        res["moves"].append((int((dP[np.isfinite(dP)] > 0).sum()), int(np.isfinite(dP).sum())))
    return m, res


def _spells(lab):
    s = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    return list(zip(s, np.r_[s[1:], len(lab)], lab[s]))


def _pixel_scale(m):
    """px per metre between neighbouring hand-annotated keypoints on the touchlines and the halfway line."""
    kp = json.load(open(stk._raw(m, "keypoints.json"), encoding="utf-8"))
    pts = {tuple(float(v) for v in re.findall(r"[-\d.]+", k)): np.array(p, float) for k, p in kp.items()}
    rows = {line: np.mean([p[1] for (x, y), p in pts.items() if y == line]) for line in (0.0, 68.0)}
    near = max(rows, key=rows.get)                  # the touchline lower in the image is the near one
    res = {}
    for line in (0.0, 68.0):
        xs = sorted(x for x, y in pts if y == line)
        for x0, x1 in zip(xs, xs[1:]):
            xm = (x0 + x1) / 2
            part = "middle third" if 35 <= xm <= 70 else "corner" if xm <= 16.5 or xm >= 88.5 else "in between"
            res.setdefault(("near" if line == near else "far") + " touchline, " + part, []).append(
                np.hypot(*(pts[(x1, line)] - pts[(x0, line)])) / (x1 - x0))
    return res


def measure():
    from multiprocessing import Pool
    mts = stk.matches()
    with Pool(min(5, len(mts))) as pool:
        res = dict(pool.map(_measure_one, mts))
    n = len(mts)
    print("SoccerTrack v2: %d matches with the provider's ball status, %d halves (%s carry none)" % (
        n, 2 * n, " ".join(stk.NO_STATUS)))
    tot = np.sum([s for r in res.values() for s in r["status"]], 0)
    print("1. provider status, share of frames: HOME %.3f AWAY %.3f BALLOUT %.3f NEUTRAL %.3f" % tuple(tot / tot.sum()))
    for c in CONVS:
        halves = [lab for m in mts for lab in res[m]["labels"][c]]
        s = ms.spell_stats(halves)
        lens = np.concatenate([[e - a for a, e, k in _spells(lab) if k < 2] for lab in halves])
        inplay = sum(1 for lab in halves for a, e, k in _spells(lab)
                     if k < 2 and e - a < es.SHORT and 0 < a and e < len(lab) and lab[a - 1] == 1 - k and lab[e] == 1 - k)
        print("   %-7s possessions / match %4.0f | < 4 s %4.1f%% | in play between two opponent possessions, < 4 s: "
              "%4.1f / match | %.2f per alive min | alive %.1f min / match" % (
                  c, s["n_spells"] / n, 100 * np.mean(lens < es.SHORT), inplay / n, s["n_spells"] / s["alive_min"],
                  s["alive_min"] / n))
    print("   target convention elsewhere: Metrica ctrl 231 / 20% / 12, DFL evctrl 229 / 18% / 13, SC evctrl 231 / "
          "24% / 15 (per match)")
    ag = np.concatenate([r["agree"] for r in res.values()])
    print("2. events whose team is the provider status at their frame (alive): %.3f of %d" % (ag.mean(), len(ag)))
    an = np.concatenate([r["anchor"] for r in res.values()])
    k = np.sum([x for r in res.values() for x in r["kink"]], 0)
    print("3. ball - acting player at the event frame: median %.2f m, < 1.5 m %.3f, < 3 m %.3f (%d events)" % (
        np.median(an), np.mean(an < 1.5), np.mean(an < 3), len(an)))
    print("   ball turns > 30 deg (0.2 s windows, moving ball): at event frames %.3f, > 2 frames from any event %.3f; "
          "%.3f of all turns lie within 2 frames of an event" % (k[0] / k[1], k[2] / k[3], k[4] / k[5]))
    npl = np.sum([x for r in res.values() for x in r["npl"]], 0)
    mv = np.sum([x for r in res.values() for x in r["moves"]], 0)
    print("4. players per alive frame: 22 in %.3f, 20-21 in %.3f, other %.3f; a player's position changes between "
          "consecutive frames in %.3f of player-frames (grid 1.05 x 0.68 m)" % (
              npl[22] / npl.sum(), npl[20:22].sum() / npl.sum(), 1 - npl[20:23].sum() / npl.sum(), mv[0] / mv[1]))
    agg = {}
    for m in sorted(d for d in os.listdir(os.path.join(stk.DATA, "raw")) if d.isdigit()):
        for key, v in _pixel_scale(m).items():
            agg.setdefault(key, []).extend(v)
    print("5. video px per metre along the touchlines, median [p10, p90] over the ten calibrations (BAR-ATH tactical "
          "camera: 1920 px over 52-76 m, ~25-37 px/m):")
    for key in sorted(agg):
        v = np.array(agg[key])
        print("   %-34s %5.1f [%5.1f, %5.1f]" % (key, np.median(v), np.percentile(v, 10), np.percentile(v, 90)))


# -- eval ---------------------------------------------------------------------------------------------

def v10_raw(k):
    """v10's full-data team classifier of seed k (DFL native + Metrica ctrl, W_M 2) on SoccerTrack."""
    path = os.path.join(OUT, "v10_seed%d.pkl" % k)
    if os.path.isfile(path):
        return bench._load(path)
    import exp_combo
    mdl = bench.load_booster(os.path.join(bench.CACHE, "combo", "a2_bfree_all%s_wm2.txt" % (
        "" if k == 0 else "_seed%d" % k)))
    bench.features(model_v7.VERSION, SRC)
    res = [bench.predict_team(mdl, f) for f in exp_combo.aug_feats(SRC)]
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def gru_mean():
    """The shared BiGRU's three members (models/bigru_dfl_pff_v1, as src/exp_shared_blend.py runs them), equal mean."""
    path = os.path.join(OUT, "bigru_members.pkl")
    if not os.path.isfile(path):
        import torch
        import exp_ctrl2 as e2
        import seq_pff as seq
        models = os.path.join(bench.ROOT, "models", "bigru_dfl_pff_v1")
        manifest = json.load(open(os.path.join(models, "manifest.json"), encoding="utf-8"))
        nets = []
        for item in manifest["models"]:
            ck = torch.load(os.path.join(models, item["file"]), map_location="cpu", weights_only=False)
            assert (ck["step"], ck["core"], ck["halo"]) == (seq.STEP, seq.CORE, seq.HALO)
            net = seq.Net(2 * len(manifest["feature_names"]) + 1, ck["config"]["kind"], ck["config"]["width"],
                          ck["config"]["drop"])
            net.load_state_dict(ck["state"])
            nets.append((net.eval(), seq.Scaler(ck["mean"], ck["scale"])))
        bench.features(model_v7.VERSION, SRC)
        members = [[] for _ in nets]
        for i, b in enumerate(bench.blocks(SRC)):
            f = bench._load(e2._feat_path(b))
            assert f["fk"] + f["tk"] == manifest["feature_names"]
            raw = seq.views(f)
            for pred, (net, scaler) in zip(members, nets):
                pred.append(seq.predict_raw(net, scaler, raw, len(b["P"])))
            print("  BiGRU %d / %d halves" % (i + 1, len(bench.blocks(SRC))), flush=True)
        os.makedirs(OUT, exist_ok=True)
        bench._dump({"members": members}, path)
    m = bench._load(path)["members"]
    return [np.mean([x[i] for x in m], axis=0) for i in range(len(m[0]))]


def c1_outputs():
    path = os.path.join(ss.OUT, "C1_%s.pkl" % SRC.replace("@", "-"))
    return bench._load(path) if os.path.isfile(path) else ss.infer(SRC, "C1")


def verify(name, blocks, preds, smos, raws, grus, c1):
    """The shipped verifier of name on decoded halves."""
    import model_v15
    import short_verifier as sv
    min_len = int(model_v7.DEFAULT["min_poss"])
    if name == "v14":
        import model_v14
        cfg = model_v14.VERIFIER
        models = sv.train(cfg["use_g"], base="blend", w=cfg["w"])
        keys = sv.keys(cfg["use_g"])
    else:
        cfg = dict(model_v15.VERIFIER, use_ins=True, use_del=True)
        models = model_v15.train_verifier()
        keys = model_v15.KEYS
    out = []
    for i, b in enumerate(blocks):
        pr = np.asarray(preds[i]).astype(np.int8)
        extra = None
        if name == "v15":
            o = np.asarray(c1[i], np.float32)
            fav = np.where(ea.logit(smos[i]) > 0, 0, 1)
            st = np.flatnonzero(np.r_[True, pr[1:] != pr[:-1]])
            own = [(a + s0, a + t0, 1 - kk) for a, e, kk in zip(st, np.r_[st[1:], len(pr)], pr[st])
                   for s0, t0 in ea.runs(fav[a:e] != kk) if kk < 2 and t0 - s0 >= ea.RUN]
            extra = [ss.kpu_cands(o, pr, ss.contest_mask(b), model_v15.THETA, own)]
        rows = ev.rows_for(SRC, [b], [ea.frame_cues(b)], [pr], [pr], [smos[i]], [raws[i]], [grus[i]], min_len,
                           extra=extra)
        if not rows:
            out.append(pr)
            continue
        if name == "v15":
            ss.touch_features(ss.add_features(rows, [o]), [o])
        p = np.full(len(rows), np.nan)
        for typ in ("ins", "del"):
            idx = [n for n, r in enumerate(rows) if r["typ"] == typ]
            if idx:
                p[idx] = models[typ].predict(np.array([[rows[n][k] for k in keys] for n in idx], float))
        out.append(ev.apply(pr, rows, p, cfg["use_ins"], cfg["use_del"], cfg["t_ins"], cfg["t_del"]))
    return out


def evaluate():
    import exp_refine as er
    blocks = bench.blocks(SRC)
    labs = {c: bench.labels(SRC, c) for c in CONVS}
    gru = gru_mean()
    c1 = c1_outputs()
    learned = bench.dead_ext("base", "v3", SRC)
    deads = (("learned D", learned), ("true D", [np.where(g == 2, 1 - es.EPS, es.EPS) for g in labs["evctrl"]]))
    cells = {}
    for k in es.SEEDS:
        raw14 = es.blend(v10_raw(k), gru, 0.5)
        raws = {"v14": raw14,
                "v15": [0.75 * r + 0.25 * ss.pk(o, 0) for r, o in zip(raw14, c1[ss.SEEDS[k]])]}
        for name in NAMES:
            smo = exp_smooth.smooth(raws[name], 25)
            for dname, pds in deads:
                final = verify(name, blocks, es.decode(DEC, 48.0, pds, smo), smo, raws[name], gru,
                               [np.asarray(o, np.float32) for o in c1[ss.SEEDS[k]]])
                for c in CONVS:
                    cells.setdefault((SRC, c, dname, name), []).append(er.by_match(blocks, labs[c], final))
        print("  seed %d done" % k, flush=True)
    er.summarize(cells, list(NAMES), ref=NAMES[0])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    {"measure": measure, "eval": evaluate}[sys.argv[1]]()
