"""Round 23, gate 1 of the short-possession specialist (src/short_specialist.py; docs/RAPOR.md Tur 23 pre-registration).
The specialist never saw DFL, Metrica or SkillCorner, so its outputs on them are out of sample. AGGREGATE output only.

  (a) the verifier's leave-one-match-out AUC with the specialist's five candidate features added to G1 (v14's rows,
      both dead channels pooled, three v10 seeds each paired with one specialist seed, verifier seed 0), per insert /
      delete x source, against G1 trained the same way. Pass: mean gain over the six cells >= +0.02, no cell < -0.01.
  (b) coverage of true in-play short spells (< 4 s between two opponent spells, IoU >= 0.3) by "insert" candidates:
      v14's (runs >= 12 frames where the +-25 blend favours the other team) against the specialist's (runs >= 6 frames
      where its live-normalised control probability of the other team exceeds 0.5), inside v14's decoded possessions.
      Pass: >= 1.5x v14's coverage with no more candidates per half (pooled over the three sources).

  python -I src/exp_kpu.py infer VARIANT          the specialist on DFL / Metrica / SkillCorner (3 seeds, CPU)
  python -I src/exp_kpu.py gate1 VARIANT          (a) and (b)
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import exp_sb  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_smooth  # noqa: E402
import exp_verify as ev  # noqa: E402
import short_specialist as ss  # noqa: E402

KPU = ["kpu_c", "kpu_cmax", "kpu_gin", "kpu_gout", "kpu_ctx"]
SRCS = [src for src, *_ in ev.SOURCES]


def outputs(variant, src):
    return bench._load(os.path.join(ss.OUT, "%s_%s.pkl" % (variant, src.replace("@", "-"))))


def pk(o, k):
    """Live-normalised probability that team k controls the ball, per frame."""
    o = np.asarray(o, np.float32)
    return o[:, k] / np.maximum(o[:, 0] + o[:, 1], 1e-6)


def add_features(rows, outs):
    for r in rows:
        o = np.asarray(outs[r["i"]], np.float32)
        n, s, t, k = len(o), r["s"], r["t"], r["k"]
        p = pk(o, k)
        r["kpu_c"] = float(p[s:t].mean())
        r["kpu_cmax"] = float(p[s:t].max())
        r["kpu_gin"] = float(o[max(0, s - 12):min(n, s + 13), 3 + k].max())
        r["kpu_gout"] = float(o[max(0, t - 12):min(n, t + 13), 5 + k].max())
        ctx = np.r_[p[max(0, s - 50):s], p[t:min(n, t + 50)]]
        r["kpu_ctx"] = float(ctx.mean()) if len(ctx) else 0.5
    return rows


def gate_a(variant):
    exp_sb.use_variant("v14")
    outs = {src: outputs(variant, src) for src in SRCS}
    params = dict(ev.PARAMS, seed=0, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "8")))
    auc = {}
    for k, seed in enumerate(ss.SEEDS):
        rows = []
        for dname in ev.DEADS:
            for src in SRCS:
                rows += add_features([dict(r) for r in ev._rows4(src, k, dname)["rows"]], outs[src][seed])
        for name, keys in (("G1", ev.FEATSETS["G1"]), ("G1+KPU", ev.FEATSETS["G1"] + KPU)):
            p = ev.predict_lomo(rows, keys, params)
            for typ in ("ins", "del"):
                for src in SRCS:
                    sel = np.array([r["typ"] == typ and r["src"] == src and r["y"] >= 0 for r in rows])
                    yy = np.array([r["y"] for r in rows])
                    auc.setdefault((name, typ, src), []).append(ea.auc(p[sel & (yy == 1)], p[sel & (yy == 0)]))
        print("  v10 seed %d done (%d rows)" % (k, len(rows)), flush=True)
    gains = []
    for typ in ("ins", "del"):
        for src in SRCS:
            a, b = np.mean(auc[("G1", typ, src)]), np.mean(auc[("G1+KPU", typ, src)])
            gains.append(b - a)
            print("(a) %s %-8s AUC G1 %.3f -> G1+KPU %.3f (%+.3f)" % (typ, src.split("@")[0], a, b, b - a))
    ok = np.mean(gains) >= 0.02 and min(gains) >= -0.01
    print("(a) mean gain %+.4f, worst cell %+.4f -> %s" % (np.mean(gains), min(gains), "PASS" if ok else "fail"))
    return ok


def runs(mask, min_len):
    st = np.flatnonzero(np.r_[mask[0], mask[1:] & ~mask[:-1]])
    en = np.flatnonzero(np.r_[mask[:-1] & ~mask[1:], mask[-1]]) + 1
    return [(a, b) for a, b in zip(st, en) if b - a >= min_len]


def gate_b(variant):
    exp_sb.use_variant("v14")
    tot = {"short": 0, "v14": 0, "kpu": 0, "union": 0, "n_v14": 0, "n_kpu": 0, "halves": 0}
    for src, convs, dec, prim in ev.SOURCES:
        labs = bench.labels(src, prim)
        outs = outputs(variant, src)
        members = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
        per = {"short": 0, "v14": 0, "kpu": 0, "union": 0, "n_v14": 0, "n_kpu": 0, "halves": 0}
        for k, seed in enumerate(ss.SEEDS):
            smo = exp_smooth.smooth(es.blend(es.v10_raw(k, src), members[k], 0.5), 25)
            for dname in ev.DEADS:
                base = ev._rows4(src, k, dname)["base"]
                for i, (g, pr) in enumerate(zip(labs, base)):
                    g = np.asarray(g)
                    st = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
                    sp = list(zip(st, np.r_[st[1:], len(g)], g[st]))
                    short = [(a, e, kk) for j, (a, e, kk) in enumerate(sp) if kk < 2 and e - a < ea.SHORT
                             and 0 < j < len(sp) - 1 and sp[j - 1][2] == 1 - kk and sp[j + 1][2] == 1 - kk]
                    fav = np.where(ea.logit(smo[i]) > 0, 0, 1)
                    o = np.asarray(outs[seed][i], np.float32)
                    c14 = [(s, t, 1 - kk) for kk in (0, 1) for s, t in runs((pr == kk) & (fav != kk), ea.RUN)]
                    ck = [(s, t, 1 - kk) for kk in (0, 1) for s, t in runs((pr == kk) & (pk(o, 1 - kk) > 0.5), 6)]

                    def hit(a, e, kk, cands):
                        return any(c == kk and min(e, t) - max(a, s) >= 0.3 * (max(e, t) - min(a, s))
                                   for s, t, c in cands)
                    for a, e, kk in short:
                        h14, hk = hit(a, e, kk, c14), hit(a, e, kk, ck)
                        per["v14"] += h14
                        per["kpu"] += hk
                        per["union"] += h14 or hk
                    per["short"] += len(short)
                    per["n_v14"] += len(c14)
                    per["n_kpu"] += len(ck)
                    per["halves"] += 1
        print("(b) %-8s in-play short spells %5d | covered v14 %.3f, specialist %.3f, union %.3f | candidates per half "
              "v14 %.1f, specialist %.1f" % (src.split("@")[0], per["short"], per["v14"] / per["short"],
                                            per["kpu"] / per["short"], per["union"] / per["short"],
                                            per["n_v14"] / per["halves"], per["n_kpu"] / per["halves"]), flush=True)
        for key in tot:
            tot[key] += per[key]
        bench._blocks.clear()
        bench._labels.clear()
    r14, rk = tot["v14"] / tot["short"], tot["kpu"] / tot["short"]
    n14, nk = tot["n_v14"] / tot["halves"], tot["n_kpu"] / tot["halves"]
    ok = rk >= 1.5 * r14 and nk <= n14
    print("(b) pooled: coverage v14 %.3f, specialist %.3f (x%.2f), union %.3f | candidates per half v14 %.1f, specialist "
          "%.1f -> %s" % (r14, rk, rk / max(r14, 1e-9), tot["union"] / tot["short"], n14, nk, "PASS" if ok else "fail"))
    return ok


THETAS = (0.6, 0.7, 0.8, 0.9)
BUDGET = 25.0                                   # candidates per half, about v14's own (23)


def contest_mask(b):
    """Frames where both teams have a player within 3 m of the ball (the specialist's own representation)."""
    x = ss.features(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
    c = 8 + 14 * ss.K
    return (x[:, 0, c] > 0) & (x[:, 0, c + 1] > 0)


def kpu_cands(o, pr, contest, theta, existing):
    """Insert candidates of the specialist: runs >= 6 frames inside a decoded possession of team k where its live-
    normalised control probability of the other team exceeds theta, touching a contested frame, not overlapping one of
    v14's own insert candidates of that team (IoU >= 0.3)."""
    out = []
    for kk in (0, 1):
        for s, t in runs((pr == kk) & (pk(o, 1 - kk) > theta), 6):
            if not contest[s:t].any():
                continue
            if any(c == 1 - kk and min(t, e) - max(s, a) >= 0.3 * (max(t, e) - min(s, a)) for a, e, c in existing):
                continue
            out.append((s, t, 1 - kk))
    return out


def theta(variant):
    """Label-free choice (pre-registered): the smallest theta with <= BUDGET specialist candidates per half, v10 seed
    0, learned dead ball, the three sources together."""
    exp_sb.use_variant("v14")
    n = {th: 0 for th in THETAS}
    halves = 0
    for src in SRCS:
        outs = outputs(variant, src)[ss.SEEDS[0]]
        d = ev._rows4(src, 0, ev.DEADS[0])
        for i, b in enumerate(bench.blocks(src)):
            con = contest_mask(b)
            ex = [(r["s"], r["t"], r["k"]) for r in d["rows"] if r["i"] == i and r["typ"] == "ins"]
            for th in THETAS:
                n[th] += len(kpu_cands(np.asarray(outs[i], np.float32), d["base"][i], con, th, ex))
            halves += 1
        bench._blocks.clear()
    per = {th: n[th] / halves for th in THETAS}
    print("specialist candidates per half: " + ", ".join("theta %.1f: %.1f" % kv for kv in per.items()))
    chosen = next((th for th in THETAS if per[th] <= BUDGET), THETAS[-1])
    print("chosen theta %.1f" % chosen)
    return chosen


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    stage, variant = sys.argv[1], sys.argv[2]
    if stage == "infer":
        for src in SRCS:
            ss.infer(src, variant)
            bench._blocks.clear()
    elif stage == "theta":
        theta(variant)
    elif stage == "gate1":
        a = gate_a(variant)
        b = gate_b(variant)
        print("gate 1 %s: %s" % (variant, "PASS" if (a or b) else "FAIL"))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
