"""How good is the Smart Tagging as possession evidence? (ideas F1 / F2 in docs/RAPOR.md)
AGGREGATE output only: counts, agreement rates and offset quantiles; no tag rows are printed.

  1. phase tags (team in possession / defending team) vs the public GT and vs v5, per code family
  2. set-piece tags vs restarts (D -> alive): offset of the tag start to the nearest restart and whether
     the tagged team is the restarting team -- public GT and v5 on the full match
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402
import evaluate as ev  # noqa: E402
import features as F  # noqa: E402
import model_v5  # noqa: E402
from make_submission import merged_features  # noqa: E402

PUB = (11185, 18685)   # public GT window (first 5 minutes of the 1st half, video frames)


def implied_team(T, n, codes_att, codes_def):
    """Per frame: tag-implied team in possession (0/1), -1 none, -2 conflicting tags."""
    vote = np.zeros((n, 2), np.int32)
    for r in T.itertuples(index=False):
        if r.team < 0:
            continue
        if r.code in codes_att:
            k = r.team
        elif r.code in codes_def:
            k = 1 - r.team
        else:
            continue
        vote[max(0, r.sf):min(n, r.ef), k] += 1
    out = np.full(n, -1, np.int8)
    out[(vote[:, 0] > 0) & (vote[:, 1] == 0)] = 0
    out[(vote[:, 1] > 0) & (vote[:, 0] == 0)] = 1
    out[(vote[:, 0] > 0) & (vote[:, 1] > 0)] = -2
    return out


def restarts(lab, lo, hi):
    seg = lab[lo:hi]
    idx = np.flatnonzero((seg[:-1] == 2) & (seg[1:] < 2)) + 1
    return idx + lo, seg[idx]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    T = F.load_tags()
    n = atd.N_FRAMES["tactical"]
    print("tag instances %d; by code (count, with team): " % len(T) + ", ".join(
        "%s %d/%d" % (c, len(g), int((g.team >= 0).sum())) for c, g in T.groupby("code")))
    sp = T[T.code == "SET PIECES"]
    print("set-piece types: " + ", ".join("%s %d" % (t or "-", len(g)) for t, g in sp.groupby("type")))
    dur = (T.ef - T.sf) / atd.FPS
    print("tag duration s: median %.1f p10 %.1f p90 %.1f" % (np.median(dur), np.percentile(dur, 10),
                                                           np.percentile(dur, 90)))

    gt = ev.public_gt()
    f = merged_features()
    v5 = model_v5.predict(f)
    fam = {"proxy (BUILD UP, PROGRESSION, LONG BALL, ATT. TRANS. / DEF. x3)": (
        ("BUILD UP", "PROGRESSION", "LONG BALL", "ATTACKING TRANSITION"),
        ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENSIVE TRANSITION")),
           "all attack / defend codes": (F.ATTACK_CODES, F.DEFEND_CODES)}
    pub = np.zeros(n, bool)
    pub[PUB[0]:PUB[1]] = True
    graded = atd.graded_mask(n)
    for name, (ca, cd) in fam.items():
        imp = implied_team(T, n, ca, cd)
        alive_gt = pub & (gt < 2)
        cov = alive_gt & (imp >= 0)
        print("\n[%s]" % name)
        print("  public GT-alive frames %d, covered by one-team tags %.3f, conflicting %.3f" % (
            alive_gt.sum(), cov.sum() / alive_gt.sum(), np.sum(alive_gt & (imp == -2)) / alive_gt.sum()))
        print("  public: tag team == GT team %.4f | v5 team == GT team on the same frames %.4f | "
              "v5 == tag %.4f" % (np.mean(imp[cov] == gt[cov]), np.mean(v5[cov] == gt[cov]),
                                  np.mean(v5[cov] == imp[cov])))
        both = cov & (v5 < 2)
        agree = both & (v5 == imp)
        print("  public: GT team when v5 and tag agree %.4f (share %.3f) / disagree: tag right %.4f, v5 right "
              "%.4f (n %d)" % (np.mean(gt[agree] == v5[agree]), agree.sum() / max(1, both.sum()),
                               np.mean(gt[both & ~agree] == imp[both & ~agree]),
                               np.mean(gt[both & ~agree] == v5[both & ~agree]), int((both & ~agree).sum())))
        full = graded & ~pub & (v5 < 2)
        cv_ = full & (imp >= 0)
        print("  full match outside public, v5-alive frames: covered %.3f, v5 == tag %.4f" % (
            cv_.sum() / full.sum(), np.mean(v5[cv_] == imp[cv_])))
        for code in sorted(set(ca) | set(cd)):
            one = implied_team(T[T.code == code], n, ca, cd)
            m = full & (one >= 0)
            if m.sum():
                print("    %-30s frames %6d  v5 == tag %.4f" % (code, m.sum(), np.mean(v5[m] == one[m])))

    # set pieces vs restarts
    print("\n[set pieces vs restarts]")
    for name, lab, lo, hi, excl in (("public GT", gt, PUB[0], PUB[1], False), ("v5 full match", v5, 0, n, True)):
        rs, rteam = restarts(lab, lo, hi)
        offs, same, n_sp = [], [], 0
        for r in sp.itertuples(index=False):
            if not (lo <= r.sf < hi) or (excl and PUB[0] <= r.sf < PUB[1]) or not graded[min(r.sf, n - 1)]:
                continue
            n_sp += 1
            if not len(rs):
                continue
            j = np.argmin(np.abs(rs - r.sf))
            offs.append(rs[j] - r.sf)
            if r.team >= 0 and abs(rs[j] - r.sf) <= 125:
                same.append(rteam[j] == r.team)
        offs = np.array(offs) / atd.FPS
        if len(offs):
            print("  %-14s set pieces %d, restarts %d | nearest restart - tag start (s): median %+.2f, p25 %+.2f, "
                  "p75 %+.2f, within 2 s %.3f, within 5 s %.3f | restart team == tag team (within 5 s) %.3f (n %d)" % (
                      name, n_sp, len(rs), np.median(offs), np.percentile(offs, 25), np.percentile(offs, 75),
                      np.mean(np.abs(offs) <= 2), np.mean(np.abs(offs) <= 5), np.mean(same) if same else np.nan,
                      len(same)))
        else:
            print("  %-14s set pieces %d, restarts %d" % (name, n_sp, len(rs)))


if __name__ == "__main__":
    main()
