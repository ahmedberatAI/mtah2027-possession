"""Round 8: v7 + capped PASS-anchor evidence from the video pilot on BAR-ATH. Anchors = PASS peaks >= MIN_CONF of
cache/video/bas_raw_fold5.npz (colab/bas_pilot.py) on the tactical timeline, team of the player nearest to the ball
OFFSET frames before the peak (<= 3 m, src/bas_eval.py); fusion as the event-anchor oracle (src/exp_anchor.py:
W0 / W1 window, clipped, beta). Written in model_v7's probability format (cache/video/barath_probs_anchor_*.pkl);
label-free indicators plus the public window for information only. AGGREGATE output only.

  python -I src/anchor_barath.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402
import bas_eval as be  # noqa: E402
import bench  # noqa: E402
import broadcast  # noqa: E402
import ctrl_barath  # noqa: E402
import exp_anchor as ea  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import model_v7  # noqa: E402

MIN_CONF = 0.35
OFFSET = -6
BETAS = (2.0, 4.0)


def anchors(fold=5):
    """(m, 2) tactical frame + team (0 home / 1 away) of the attributed PASS events."""
    z = np.load(os.path.join(be.VIDEO, "bas_raw_fold%d.npz" % fold))
    t2b = broadcast.tactical_to_broadcast()
    b2t = np.full(atd.N_FRAMES["broadcast"] + 1, -1, np.int64)
    ok = t2b >= 0
    b2t[t2b[ok]] = np.flatnonzero(ok)
    fr, conf = z["pass_frames"], z["pass_conf"]
    tt = b2t[np.clip(fr[conf >= MIN_CONF], 0, len(b2t) - 1)]
    tt = tt[(tt >= 0) & atd.graded_mask()[np.maximum(tt, 0)]]
    blocks = bench.blocks("barath")
    out = []
    for t in tt:
        k = be.nearest_team(blocks, int(t + OFFSET))[0]
        if k >= 0:
            out.append((int(t + OFFSET), k))
    return np.array(out, np.int64)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    from make_submission import merged_features
    f = merged_features()
    anc = anchors()
    print("attributed PASS anchors: %d" % len(anc), flush=True)
    for clean in (False, True):
        tag = "cleanball" if clean else "asis"
        base_path = model_v7.ensure_probs(clean_ball=clean)
        probs = bench._load(base_path)
        ref = ctrl_barath.indicators("v7 " + tag, f, base_path)
        for beta in BETAS:
            out = {}
            for a, (_, p) in probs.items():
                n = len(p)
                loc = anc[(anc[:, 0] >= a) & (anc[:, 0] < a + n)].copy()
                loc[:, 0] -= a
                out[a] = (None, e2.sig(e2.logit(p) + beta * ea.evidence(loc, n)))
            path = os.path.join(be.VIDEO, "barath_probs_anchor_b%g_%s.pkl" % (beta, tag))
            bench._dump(out, path)
            ctrl_barath.indicators("anchors b%g %s" % (beta, tag), f, path, ref)


if __name__ == "__main__":
    main()
