"""Is a per-frame whistle score discriminative for referee whistles? Aggregate statistics only.

    .venv/Scripts/python.exe src/whistle_eval.py sed:Whistle clip32:Whistle@exc naive:tonal_ex ...

Score spec = <source>:<class>[@exc]. source = a whistle_panns variant (sed | clip16 | clip32,
score = logit of the class probability; "A-B" = logit A minus logit B), `naive` (src/whistle.py,
class = one of its arrays) or `ref` (src/whistle_ref.py, class = one of its arrays).
"@exc" subtracts a +-30 s rolling median (crowd baseline).

  kickoff  max score in the 3 s before each kickoff (the whistle precedes the first touch by
           ~1.1 s): offset, percentile among graded frames, rank among the match's peaks
  gt       public-GT D starts / D ends: mean robust z-score in lag bins, and how many events have
           one of the top-N peaks within [-2, +1] s
  v5       hit rate of the top-N peaks near the D starts / D ends of submissions/baseline_v5.csv
           versus the same window placed at every graded frame and at every predicted-alive frame
           (exact expectations), binomial p-value, share of peaks near any predicted boundary and
           a histogram of peak offsets relative to the nearest predicted D start
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from scipy.ndimage import maximum_filter1d  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

import atd  # noqa: E402
import evaluate as ev  # noqa: E402

pm = ev.pm
N = atd.N_FRAMES["tactical"]
FPS = atd.FPS
GRADED = atd.graded_mask(N)
KICKOFFS = (atd.HALVES[0][1], atd.KICKOFF_2ND)
V5 = os.path.join(ROOT, "submissions", "baseline_v5.csv")
TOPN = (50, 100, 200)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def get_score(spec):
    import whistle
    src, _, rest = spec.partition(":")
    cls, _, mode = rest.partition("@")
    if src == "naive":
        s = whistle.compute()[cls or "tonal_ex"].astype(np.float64)
    elif src == "ref":
        import whistle_ref
        s = whistle_ref.build()[cls or "s"].astype(np.float64)
    else:
        import whistle_panns as wp
        if "-" in cls:
            a, b = cls.split("-")
            s = logit(wp.frame_scores(src, a)) - logit(wp.frame_scores(src, b))
        else:
            s = logit(wp.frame_scores(src, cls))
    if mode == "exc":
        s = s - whistle.rolling_median(s.astype(np.float32), 750)
    return s.astype(np.float64)


def robust_z(s):
    g = s[GRADED]
    med = np.median(g)
    mad = 1.4826 * np.median(np.abs(g - med)) or g.std() or 1.0
    return (s - med) / mad


def peaks(s, mask, radius=25):
    """Local maxima (non-maximum suppression over +-radius frames) inside mask, highest first."""
    mx = maximum_filter1d(s, 2 * radius + 1, mode="nearest")
    cand = np.flatnonzero((s >= mx) & mask)
    cand = cand[np.argsort(-s[cand], kind="stable")]
    taken = np.zeros(len(s), bool)
    keep = []
    for i in cand:
        if not taken[max(0, i - radius):i + radius + 1].any():
            keep.append(i)
            taken[i] = True
    return np.array(keep, dtype=np.int64)


def transitions(codes, mask=GRADED):
    alive = (codes == pm.H) | (codes == pm.A)
    ok = mask[1:] & mask[:-1]
    starts = np.flatnonzero(ok & (codes[1:] == pm.D) & alive[:-1]) + 1
    ends = np.flatnonzero(ok & alive[1:] & (codes[:-1] == pm.D)) + 1
    return starts, ends


def window_hits(events, pk, a, b):
    """Per event: does a peak fall in [t + a, t + b] (frames)?"""
    m = np.zeros(N + 1, np.int64)
    m[pk] = 1
    cs = np.concatenate(([0], np.cumsum(m)))
    lo = np.clip(np.asarray(events) + a, 0, N)
    hi = np.clip(np.asarray(events) + b + 1, 0, N)
    return (cs[hi] - cs[lo]) > 0


def nearest_offset(pk, events):
    """Signed offset (frames) of each peak to the nearest event."""
    ev_ = np.sort(np.asarray(events))
    i = np.clip(np.searchsorted(ev_, pk), 1, len(ev_) - 1)
    left, right = ev_[i - 1], ev_[i]
    return np.where(np.abs(pk - left) <= np.abs(pk - right), pk - left, pk - right)


def report(spec, v5=None):
    s = get_score(spec)
    z = robust_z(s)
    pk_all = peaks(s, GRADED)
    out = ["=" * 100, "%s   (%d peaks in graded frames)" % (spec, len(pk_all))]

    # ---- kickoffs
    parts = []
    for k in KICKOFFS:
        lo = k - 75
        j = lo + int(np.argmax(s[lo:k + 6]))
        pct = 100.0 * (s[GRADED] < s[j]).mean()
        rank = 1 + int((s[pk_all] > s[j]).sum())
        parts.append("kickoff %d: peak %+.2f s, z %.1f, pct %.2f, rank %d" % (k, (j - k) / FPS, z[j], pct, rank))
    out.append("  " + " | ".join(parts))

    # ---- public GT boundaries
    gt = ev.public_gt()
    win = np.zeros(N, bool)
    win[11185:18685] = True
    gs, ge = transitions(gt, win)
    bins = [(-5, -3), (-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 5)]
    for name, evs in (("GT D start", gs), ("GT D end  ", ge)):
        row = []
        for a, b in bins:
            vals = [z[t + a * FPS:t + b * FPS].mean() for t in evs]
            row.append("%+.2f" % np.mean(vals))
        hits = {n: int(window_hits(evs, pk_all[:n], -2 * FPS, FPS).sum()) for n in TOPN}
        out.append("  %s n=%d  mean z in %s: %s | top-N peak in [-2,+1]s: %s" % (
            name, len(evs), ",".join("%d..%d" % ab for ab in bins), " ".join(row),
            " ".join("N%d=%d" % kv for kv in hits.items())))

    # ---- v5 predicted boundaries
    if v5 is not None:
        vs, ve = transitions(v5)
        alive = GRADED & ((v5 == pm.H) | (v5 == pm.A))
        graded_idx = np.flatnonzero(GRADED)
        alive_idx = np.flatnonzero(alive)
        for name, evs, (a, b) in (("v5 D start", vs, (-3 * FPS, 1 * FPS)), ("v5 D start", vs, (-1 * FPS, 1 * FPS)),
                                  ("v5 D end  ", ve, (-4 * FPS, 1 * FPS))):
            cells = []
            for n in TOPN:
                pk = pk_all[:n]
                h = window_hits(evs, pk, a, b)
                p_all = window_hits(graded_idx, pk, a, b).mean()
                p_alive = window_hits(alive_idx, pk, a, b).mean()
                pv = binomtest(int(h.sum()), len(evs), max(p_all, p_alive), alternative="greater").pvalue
                cells.append("N%d %2d/%d=%.2f (rand %.2f/alive %.2f, lift %.1f, p=%.1e)" % (
                    n, h.sum(), len(evs), h.mean(), p_all, p_alive, h.mean() / max(p_all, p_alive, 1e-9), pv))
            out.append("  %s [%+.0f,%+.0f]s: %s" % (name, a / FPS, b / FPS, " | ".join(cells)))
        # where do the strongest peaks sit?
        for n in TOPN:
            pk = pk_all[:n]
            near_s = np.abs(nearest_offset(pk, vs) + FPS) <= 2 * FPS       # within [-3, +1] s of a D start
            near_e = (nearest_offset(pk, ve) >= -4 * FPS) & (nearest_offset(pk, ve) <= FPS)
            dmask = np.zeros(N, bool)
            for t in vs:
                dmask[max(0, t - 3 * FPS):t + FPS + 1] = True
            emask = np.zeros(N, bool)
            for t in ve:
                emask[max(0, t - 4 * FPS):t + FPS + 1] = True
            exp_s = (dmask & GRADED).sum() / GRADED.sum()
            exp_e = (emask & GRADED).sum() / GRADED.sum()
            in_d = (v5[pk] == pm.D).mean()
            out.append("  top-%d peaks: %.2f near a D start (chance %.2f), %.2f before a D end (chance %.2f), "
                       "%.2f inside predicted D (D share %.2f)" % (
                           n, near_s.mean(), exp_s, near_e.mean(), exp_e, in_d, (v5[GRADED] == pm.D).mean()))
        off = nearest_offset(pk_all[:100], vs) / FPS
        edges = np.arange(-6, 6.01, 1.0)
        cnt, _ = np.histogram(off, edges)
        out.append("  top-100 peak offset to nearest v5 D start (1 s bins %+d..%+d s): %s  (chance/bin %.1f)" % (
            edges[0], edges[-1], " ".join(str(c) for c in cnt), 100 * len(vs) * FPS / GRADED.sum()))
        out.append("  " + context(pk_all[:100], v5, vs, ve))
    print("\n".join(out))
    sys.stdout.flush()


def context(pk, v5, vs, ve):
    """Where do the peaks sit in the v5 segmentation? (counts per time bin)"""
    vs, ve = np.sort(vs), np.sort(ve)
    in_d = v5[pk] == pm.D
    a = pk[~in_d]
    nxt = vs[np.clip(np.searchsorted(vs, a), 0, len(vs) - 1)]
    to_start = np.where(nxt >= a, (nxt - a) / FPS, np.inf)
    d = pk[in_d]
    prev = vs[np.clip(np.searchsorted(vs, d, side="right") - 1, 0, len(vs) - 1)]
    since = np.where(prev <= d, (d - prev) / FPS, np.inf)
    nxe = ve[np.clip(np.searchsorted(ve, d), 0, len(ve) - 1)]
    to_end = np.where(nxe >= d, (nxe - d) / FPS, np.inf)
    e = [0, 1, 2, 3, 5, 10, np.inf]
    h = lambda x: " ".join(str(c) for c in np.histogram(x, e)[0])  # noqa: E731
    return ("peaks in v5-alive: %d, next D start in 0-1/1-2/2-3/3-5/5-10/>10 s: %s | peaks in v5-D: %d, "
            "since D start: %s, to D end: %s" % (len(a), h(to_start), len(d), h(since), h(to_end)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    v5 = ev.to_codes(pm.read_label_csv(V5)["label"].values)
    for spec in sys.argv[1:]:
        report(spec, v5)
