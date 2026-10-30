"""Round 14 diagnostic (29 Oct, user: "do we have a problem with short possessions in the latest model?").
AGGREGATE output only.

Models (held-out probabilities, V3 inputs):
  v10    three LightGBM seeds (the round-9 / round-13 out-of-fold files), +-25-frame logit smoothing
  blend  round-13 candidate: 0.25 v10 + 0.75 shared BiGRU probability (paired seeds), then +-25; Metrica and
         SkillCorner only -- the shared BiGRU trained on all seven DFL matches
Contexts: DFL-trk, DFL-evctrl, Metrica-ctrl, SC phase, SC pp; learned and true D. Decoder B2 as shipped (ramp c 48)
and, to see what the ramp suppresses, c 24 and c 0 (D0 175, pen 12 everywhere).
Per context x model x ramp (sums over seeds and halves; Metrica's three noise draws count separately):
  true spells by duration bin: n, recall over the five IoU thresholds, share never touched by a same-team prediction
  predicted spells by bin: n, precision over the five thresholds; predicted / true count
  share of the lost F1 credit (FN + FP over the five thresholds) carried by spells < 4 s; spellF1 and two oracle
  gains: insert the missed true spells < 4 s; replace the unmatched predicted spells < 4 s by the truth
Team evidence before decoding: short (< 4 s) true spells whose probability favours the true team on most frames
  (raw and +-25), and false runs (>= 0.5 s favouring the other team) inside long (>= 8 s) true spells per alive minute.
BAR-ATH, label-free: predicted spells per alive minute, median length, share < 1 / 2 / 4 s for the shipped v10 CSV and
the blend (v10 shipped model + equal BiGRU mean). No public-window score is computed or printed.

  python -I src/exp_short.py
"""
import hashlib
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import decode2  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle as eo  # noqa: E402
import exp_smooth  # noqa: E402
import exp_v7  # noqa: E402
import train_dfl as td  # noqa: E402

FPS = 25.0
EPS = 1e-4
SHORT = 100                      # 4 s
LONG = 200                       # 8 s
RUN = 12                         # 0.5 s
BINS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, np.inf))
BIN_NAMES = ["%g-%gs" % b if np.isfinite(b[1]) else ">%gs" % b[0] for b in BINS]
N_SHORT = 3                      # the first three bins are < 4 s
RAMPS = (48.0, 24.0, 0.0)
SEEDS = (0, 1, 2)
W_BIGRU = 0.75
CONTEXTS = (("dfl@v3", ("trk", "evctrl"), (32.0, 5, 25)), ("metrica@v3", ("ctrl",), (30.0, 8, 50)),
            ("sc@v3", ("phase", "pp"), (30.0, 8, 50)))
IOUS = eo.pm.SPELL_IOUS
THREADS = int(os.environ.get("SHORT_THREADS", "10"))
EXPECTED_MD5 = "ca20c5b657a395d2f7c0e310cdaabd64"


def v10_raw(seed, src):
    suffix = "" if seed == 0 else "_seed%d" % seed
    return bench._load(os.path.join(bench.CACHE, "combo", "probs_a2_bfree%s_wm2.pkl" % suffix))[src]


def bigru_members(src):
    path = os.path.join(os.path.dirname(bench.CACHE), "shared_blend_v1", src.replace("@", "-") + "_members.pkl")
    return bench._load(path)["members"]


def blend(a, b, w=None):
    """(1 - w) x a + w x b per half; w defaults to W_BIGRU (round 13's .75; model_v14 passes its own .5)."""
    w = W_BIGRU if w is None else w
    return [(1 - w) * np.asarray(x, np.float64) + w * np.asarray(y, np.float64) for x, y in zip(a, b)]


def decode(dec, c, pds, phs):
    def one(pair):
        sp = decode2.duration_spec(12.0, dec[0], dec[1], dec[2], exp_v7.ramp(175, c))
        return sp.decode(td.emissions(pair[0], np.clip(pair[1], EPS, 1 - EPS), 1.0, 1.0)).astype(np.int8)
    first = one((pds[0], phs[0]))           # compile before the threads start
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        return [first] + list(pool.map(one, zip(pds[1:], phs[1:])))


def bin_of(n):
    s = n / FPS
    return next(i for i, (lo, hi) in enumerate(BINS) if lo <= s < hi)


class Tally:
    """Spell counts by duration bin; hits are averaged over the five IoU thresholds."""

    def __init__(self):
        z = lambda: np.zeros(len(BINS))  # noqa: E731
        self.g_n, self.g_hit, self.g_unseen = z(), z(), z()
        self.p_n, self.p_hit, self.p_unseen = z(), z(), z()

    def add(self, gt, pred):
        gs, ps = eo.spells(gt), eo.spells(pred)
        gseen, pseen = eo.overlapping(gs, ps), eo.overlapping(ps, gs)
        gh, ph = np.zeros(len(gs)), np.zeros(len(ps))
        for thr in IOUS:
            mg, mp = eo.matched(gs, ps, thr)
            gh += mg
            ph += mp
        for (a, b, _), h, s in zip(gs, gh, gseen):
            i = bin_of(b - a)
            self.g_n[i] += 1
            self.g_hit[i] += h / len(IOUS)
            self.g_unseen[i] += not s
        for (a, b, _), h, s in zip(ps, ph, pseen):
            i = bin_of(b - a)
            self.p_n[i] += 1
            self.p_hit[i] += h / len(IOUS)
            self.p_unseen[i] += not s

    def short_loss_share(self):
        fn, fp = self.g_n - self.g_hit, self.p_n - self.p_hit
        return (fn[:N_SHORT].sum() + fp[:N_SHORT].sum()) / max(fn.sum() + fp.sum(), 1e-9)

    def lines(self, indent):
        g = " ".join("%s %d/%.2f/%.2f" % (nm, n, h / max(n, 1), u / max(n, 1))
                     for nm, n, h, u in zip(BIN_NAMES, self.g_n, self.g_hit, self.g_unseen))
        p = " ".join("%s %d/%.2f/%.2f" % (nm, n, h / max(n, 1), n / max(t, 1))
                     for nm, n, h, t in zip(BIN_NAMES, self.p_n, self.p_hit, self.g_n))
        short_t, short_p = self.g_n[:N_SHORT].sum(), self.p_n[:N_SHORT].sum()
        return ["%strue  (n/recall/unseen): %s" % (indent, g),
                "%spred  (n/precision/pred:true): %s" % (indent, p),
                "%s<4 s: true %d recall %.3f, pred %d precision %.3f, pred:true %.2f; share of lost credit %.1f%%" % (
                    indent, short_t, self.g_hit[:N_SHORT].sum() / max(short_t, 1), short_p,
                    self.p_hit[:N_SHORT].sum() / max(short_p, 1), short_p / max(short_t, 1),
                    100 * self.short_loss_share())]


def drop_short_spurious(pred, gt):
    out = pred.copy()
    gs, ps = eo.spells(gt), eo.spells(pred)
    _, mp = eo.matched(gs, ps, 0.3)
    for (a, b, _), m in zip(ps, mp):
        if not m and b - a < SHORT:
            out[a:b] = gt[a:b]
    return out


def runs_at_least(mask, k):
    if not mask.any():
        return 0
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return int(((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) >= k).sum())


def evidence(phs, gts):
    """Short true spells favoured on most frames; false runs inside long true spells per alive minute."""
    n_short = major = runs = alive = 0
    for ph, gt in zip(phs, gts):
        fav = np.where(np.asarray(ph) > 0.5, 0, 1)
        for a, b, k in eo.spells(gt):
            if b - a < SHORT:
                n_short += 1
                major += np.mean(fav[a:b] == k) >= 0.5
            elif b - a >= LONG:
                runs += runs_at_least(fav[a:b] != k, RUN)
        alive += int((gt < 2).sum())
    return major / max(n_short, 1), runs / max(alive / FPS / 60, 1e-9)


def length_profile(labs):
    lens = np.array([b - a for lab in labs for a, b, _ in eo.spells(lab)])
    alive_min = sum(int((lab < 2).sum()) for lab in labs) / FPS / 60
    if not len(lens):
        return "no spells"
    return "%d spells, %.2f / alive min, median %.1f s, <1 s %.0f%%, <2 s %.0f%%, <4 s %.0f%%" % (
        len(lens), len(lens) / alive_min, np.median(lens) / FPS, 100 * np.mean(lens < 25), 100 * np.mean(lens < 50),
        100 * np.mean(lens < SHORT))


def external():
    for src, convs, dec in CONTEXTS:
        t0 = time.time()
        labs = {c: bench.labels(src, c) for c in convs}
        blocks = bench.blocks(src)
        raws = {"v10": [v10_raw(s, src) for s in SEEDS]}
        if not src.startswith("dfl"):
            members = bigru_members(src)
            raws["blend"] = [blend(a, b) for a, b in zip(raws["v10"], members)]
            del members
        smooth = {m: [exp_smooth.smooth(r, 25) for r in rs] for m, rs in raws.items()}
        learned = bench.dead_probs(src, "base") if src == "dfl@v3" else bench.dead_ext("base", "v3", src)
        deads = [("learned D", learned, convs)]
        seen = {}
        for c in convs:                     # one true-D decode per distinct dead mask
            key = hashlib.md5(b"".join((g == 2).tobytes() for g in labs[c])).hexdigest()
            seen.setdefault(key, []).append(c)
        for key, cs in seen.items():
            deads.append(("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs[cs[0]]], tuple(cs)))
        print("\n=== %s (%d halves; conventions %s) ===" % (src, len(blocks), ", ".join(convs)), flush=True)
        for c in convs:
            print("  truth %-6s %s" % (c, length_profile(labs[c])))
            for m in raws:
                cells = []
                for nm, rs in (("raw", raws[m]), ("+-25", smooth[m])):
                    ev_ = np.array([evidence(r, labs[c]) for r in rs])
                    cells.append("%s: short favoured %.3f, false runs %.2f/alive min" % (nm, *ev_.mean(0)))
                print("  evidence %-6s %-5s %s" % (c, m, " | ".join(cells)), flush=True)
        for dname, pds, cs in deads:
            for m in smooth:
                for ramp in RAMPS:
                    preds = [decode(dec, ramp, pds, phs) for phs in smooth[m]]
                    for c in cs:
                        gts = labs[c]
                        tal = Tally()
                        base, ins, drop = [], [], []
                        for pr in preds:
                            for g, p in zip(gts, pr):
                                tal.add(g, p)
                            base.append(e2.score(src, c, pr).mean())
                            ins.append(e2.score(src, c, [eo.insert_short(p, g) for p, g in zip(pr, gts)]).mean())
                            drop.append(e2.score(src, c, [drop_short_spurious(p, g) for p, g in zip(pr, gts)]).mean())
                        print("  [%s %s | %s | %s | ramp c %g] spellF1 %.4f (seeds %s); oracle insert missed <4 s "
                              "%+.4f, drop spurious <4 s %+.4f" % (
                                  src.split("@")[0], c, dname, m, ramp, np.mean(base),
                                  " ".join("%.4f" % b for b in base), np.mean(ins) - np.mean(base),
                                  np.mean(drop) - np.mean(base)))
                        if ramp == RAMPS[0] and dname == "learned D":
                            print("    predicted profile: %s" % length_profile(preds[0]))
                        for line in tal.lines("    "):
                            print(line)
                        sys.stdout.flush()
                    del preds
        print("  (%s done in %.0f s)" % (src, time.time() - t0), flush=True)
        bench._blocks.clear()
        bench._labels.clear()


def barath():
    import bfree
    import evaluate as ev
    import make_submission
    import model_v7
    import model_v9
    import model_v10
    csv = os.path.join(ev.ROOT, "submissions", "candidate_v10.csv")
    with open(csv, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()
    v10 = ev.to_codes(np.loadtxt(csv, delimiter=",", skiprows=1, usecols=1, dtype=str))
    model = model_v10.team_model()
    bench._barath_inputs()
    raw = {}
    for b, f in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        raw[b["a"]] = bench.predict_team(model, dict(XF=f["XF"], XH=np.hstack([f["XH"], h]),
                                                      XA=np.hstack([f["XA"], a])))
    cached = bench._load(model_v10.PROBS)
    dv = max(float(np.max(np.abs(model_v9.smooth(raw[a]) - cached[a][1]))) for a in raw)
    big = bench._load(os.path.join(os.path.dirname(bench.CACHE), "shared_bigru", "barath_equal_probabilities.pkl"))
    probs = {a: (None, model_v9.smooth((1 - W_BIGRU) * raw[a] + W_BIGRU * big[a][1])) for a in raw}
    path = os.path.join(bench.CACHE, "short14", "barath_blend75_probs.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bench._dump(probs, path)
    f = make_submission.merged_features()
    blend_lab = model_v7.predict(f, probs_path=path)
    same = np.mean(np.asarray(model_v7.predict(f, probs_path=model_v10.PROBS)) == np.asarray(v10))
    print("\n=== BAR-ATH full match, label-free (no public score) ===")
    print("  shipped CSV MD5 matches %s: %s; recomputed v10 probabilities vs cache max diff %.1e; "
          "re-decoded v10 equals the CSV on %.4f of frames" % (EXPECTED_MD5[:8], md5 == EXPECTED_MD5, dv, same))
    for name, lab in (("v10 (shipped CSV)", v10), ("blend 0.75", blend_lab)):
        print("  %-18s %s" % (name, length_profile([np.asarray(lab, np.int8)])))
    print("  frames that differ v10 vs blend: %.2f%%" % (100 * np.mean(np.asarray(blend_lab) != np.asarray(v10))))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "barath"):
        barath()
    if which in ("all", "external"):
        external()


if __name__ == "__main__":
    main()
