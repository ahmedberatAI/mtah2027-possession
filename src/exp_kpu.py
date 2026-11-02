"""Round 23, gate 1 of the short-possession specialist (src/short_specialist.py; docs/RAPOR.md Tur 23 pre-registration).
The specialist never saw DFL, Metrica or SkillCorner, so its outputs on them are out of sample. AGGREGATE output only.

  (a) the verifier's leave-one-match-out AUC with the specialist's five candidate features added to G1 (v14's rows,
      both dead channels pooled, three v10 seeds each paired with one specialist seed, verifier seed 0), per insert /
      delete x source, against G1 trained the same way. Pass: mean gain over the six cells >= +0.02, no cell < -0.01.
  (b) coverage of true in-play short spells (< 4 s between two opponent spells, IoU >= 0.3) by "insert" candidates:
      v14's (runs >= 12 frames where the +-25 blend favours the other team) against the specialist's (runs >= 6 frames
      where its live-normalised control probability of the other team exceeds 0.5), inside v14's decoded possessions.
      Pass: >= 1.5x v14's coverage with no more candidates per half (pooled over the three sources).

Gate 2 (pre-registered): v14's decodes and candidates, (a) the specialist's features on every candidate and (b) its
own insert candidates (kpu_cands, THETA), the pooled leave-one-match-out verifier on G1 + the five features (3 v10
seeds x 2 verifier seeds), the ten contexts against v14 with the paired match bootstrap. Names: S1a / S1ab / S2a /
S2ab (variant + uses); a trailing c also snaps the final short spells' ends to the specialist's change peaks.

  python -I src/exp_kpu.py infer VARIANT          the specialist on DFL / Metrica / SkillCorner (3 seeds, CPU)
  python -I src/exp_kpu.py gate1 VARIANT          (a) and (b)
  python -I src/exp_kpu.py theta VARIANT          the label-free candidate threshold
  python -I src/exp_kpu.py build NAME             candidate rows of S1a / S1ab / S2a / S2ab
  python -I src/exp_kpu.py trainpool NAME SEED VSEED
  python -I src/exp_kpu.py eval VSEED[,VSEED] NAME,...
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


THETA = {"S1": 0.7, "S2": 0.8}                  # src/exp_kpu.py theta, recorded in docs/RAPOR.md before gate 2
_path = ev._path


def use(name):
    """v14's blend settings with exp_verify's caches tagged w50 (v14) or w50kpu<name> (rows / probs)."""
    exp_sb.use_variant("v14")
    if name == "v14":
        return
    tag = "w50kpu" + name.rstrip("c")

    def tagged(kind, *parts):
        return _path(kind + tag if kind in ("rows3", "rows4", "probs3", "probs4") else kind, *parts)
    ev._path = tagged


def build(name):
    variant, uses = name[:2], name[2:]
    use("v14")
    v14_path = ev._path
    for src, convs, dec, prim in ev.SOURCES:
        kind = "rows4" if src.startswith("dfl") else "rows3"
        outs = outputs(variant, src)
        old = {(k, d): bench._load(v14_path(kind, src, k, d)) for k in es.SEEDS for d in ev.DEADS}
        if uses == "a":
            new = {key: dict(base=v["base"], rows=add_features([dict(r) for r in v["rows"]], outs[ss.SEEDS[key[0]]]))
                   for key, v in old.items()}
        else:
            blocks = bench.blocks(src)
            labs = bench.labels(src, prim)
            dfl = src.startswith("dfl")
            members = ev.lomo_bigru() if dfl else es.bigru_members(src)
            gru = ev.gru_mean4(src) if dfl else ev.gru_mean(src)
            raws = {k: es.blend(es.v10_raw(k, src), members[k]) for k in es.SEEDS}
            smos = {k: exp_smooth.smooth(raws[k], 25) for k in es.SEEDS}
            rows = {key: [] for key in old}
            for i, b in enumerate(blocks):
                c = ea.frame_cues(b)
                con = contest_mask(b)
                for (k, d), v in old.items():
                    pr = v["base"][i]
                    ex = [(r["s"], r["t"], r["k"]) for r in v["rows"] if r["i"] == i and r["typ"] == "ins"]
                    extra = kpu_cands(np.asarray(outs[ss.SEEDS[k]][i], np.float32), pr, con, THETA[variant], ex)
                    rows[(k, d)] += ev.rows_for(src, [b], [c], [labs[i]], [pr], [smos[k][i]], [raws[k][i]], [gru[i]],
                                                int(dec[1]), i0=i, v2=True, extra=[extra])
                del c
            new = {key: dict(base=old[key]["base"], rows=add_features(rows[key], outs[ss.SEEDS[key[0]]]))
                   for key in old}
        use(name)
        for (k, d), v in new.items():
            bench._dump(v, ev._path(kind, src, k, d))
        use("v14")
        r0 = new[(0, ev.DEADS[0])]["rows"]
        print("  %s %s: rows %d (specialist candidates %d), v10 seed 0, learned D" % (
            name, src, len(r0), int(sum(r.get("kpu_cand", 0) for r in r0))), flush=True)
        bench._blocks.clear()
        bench._labels.clear()


def trainpool(name, seed, vseed):
    use(name)
    parts = {d: [r for src in SRCS for r in ev._rows4(src, seed, d)["rows"]] for d in ev.DEADS}
    rows = parts[ev.DEADS[0]] + parts[ev.DEADS[1]]
    params = dict(ev.PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "4")))
    pp = ev.predict_lomo(rows, ev.FEATSETS["G1"] + KPU, params)
    n = len(parts[ev.DEADS[0]])
    for j, d in enumerate(ev.DEADS):
        bench._dump({"G1": pp[:n] if j == 0 else pp[n:]}, ev._path("probs4", seed, d, "q%d" % vseed))
    yy = np.array([r["y"] for r in rows])
    cells = []
    for typ in ("ins", "del"):
        for src in SRCS:
            sel = np.array([r["typ"] == typ and r["src"] == src and r["y"] >= 0 for r in rows])
            cells.append("%s %s %.3f" % (typ, src.split("@")[0], ea.auc(pp[sel & (yy == 1)], pp[sel & (yy == 0)])))
    print("    %s seed %d q%d verifier AUC %s" % (name, seed, vseed, " | ".join(cells)), flush=True)


def snap(pred, o):
    """(c) Each final possession shorter than 4 s gets its start moved to the peak of its team's "gains" head and its
    end to the peak of its "loses" head within +-0.5 s, where the peak is >= 0.5 (the neighbours follow)."""
    out = pred.copy()
    n = len(out)
    st = np.flatnonzero(np.r_[True, pred[1:] != pred[:-1]])
    for a, e in zip(st, np.r_[st[1:], n]):
        k = int(pred[a])
        if k > 1 or e - a >= ea.SHORT or a == 0 or e == n:
            continue
        lo, hi = max(1, a - 12), min(e - 1, a + 13)
        if hi > lo:
            j = lo + int(np.argmax(o[lo:hi, 3 + k]))
            if o[j, 3 + k] >= 0.5:
                if j < a:
                    out[j:a] = k
                else:
                    out[a:j] = pred[a - 1]
        lo, hi = max(a + 1, e - 12), min(n - 1, e + 13)
        if hi > lo:
            j = lo + int(np.argmax(o[lo:hi, 5 + k]))
            if o[j, 5 + k] >= 0.5:
                if j > e:
                    out[e:j] = k
                else:
                    out[j:e] = pred[e]
    return out


def evaluate(vseeds, names):
    import exp_blend_weight as bw
    import exp_refine as er
    cells = {}
    names = ["v14"] + [n for n in names if n != "v14"]
    for src, convs, dec, prim in ev.SOURCES:
        use("v14")
        blocks = bench.blocks(src)
        labs = {c: bench.labels(src, c) for c in convs}
        for name in names:
            use(name)
            outs = outputs(name[:2], src) if name.endswith("c") else None
            for seed in es.SEEDS:
                for dname in ev.DEADS:
                    for vseed in vseeds:
                        _, preds = bw._verified(src, seed, dname, vseed, lambda s2: ev._rows4(s2, seed, dname), "probs4")
                        if outs is not None:
                            preds = [snap(p, np.asarray(o, np.float32)) for p, o in zip(preds, outs[ss.SEEDS[seed]])]
                        for c in convs:
                            cells.setdefault((src, c, dname, name), []).append(er.by_match(blocks, labs[c], preds))
        bench._blocks.clear()
        bench._labels.clear()
    use("v14")
    mean, regret = er.summarize(cells, names, ref="v14")
    for name in names[1:]:
        bw.regret_boot(mean, ["v14", name], "v14", name)
    return mean, regret


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    stage, variant = sys.argv[1], sys.argv[2]
    if stage == "build":
        build(variant)
    elif stage == "trainpool":
        trainpool(variant, int(sys.argv[3]), int(sys.argv[4]))
    elif stage == "eval":
        evaluate(tuple(int(x) for x in sys.argv[2].split(",")), sys.argv[3].split(","))
    elif stage == "infer":
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
