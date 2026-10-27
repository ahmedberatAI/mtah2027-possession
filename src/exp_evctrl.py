"""Round 12 (R12-3): re-score cached candidates on the new DFL context "evctrl" (controlled possession from the
synchronised DFL events, src/dfl_events.py) -- an event-based, GT-like convention (~230 spells / match). Ideas that lost
on DFL-trk may look different here. Decode-only on cached leave-one-match-out probabilities (+-25 logit smoothing,
DFL decoder), learned and true D; v10's three seeds as reference. AGGREGATE output only.

    python -I src/exp_evctrl.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import exp_smooth  # noqa: E402

C = bench.CACHE
SRC, DEC = "dfl@v3", (32.0, 5, 25)
CANDS = [
    ("v10 seed 0", "combo/probs_a2_bfree_wm2.pkl"), ("v10 seed 1", "combo/probs_a2_bfree_seed1_wm2.pkl"),
    ("v10 seed 2", "combo/probs_a2_bfree_seed2_wm2.pkl"),
    ("v9 seed 0", "combo/probs_a2_bfree.pkl"), ("v9 seed 1", "combo/probs_a2_bfree_seed1.pkl"),
    ("v9 seed 2", "combo/probs_a2_bfree_seed2.pkl"),
    ("v8 (A2, no bfree)", "a2/probs_wm1.pkl"), ("A2 wm3", "a2/probs_wm3.pkl"), ("v10 wm3", "combo/probs_a2_bfree_wm3.pkl"),
    ("v10 big LGB", "combo/probs_a2_bfree_wm2_l127_r600.pkl"), ("v10 DFL tau2", "combo/probs_a2_bfree_wm2_tau2.pkl"),
    ("v10 DFL tau0.4", "combo/probs_a2_bfree_wm2_tau04.pkl"), ("v10 norm", "combo/probs_a2_bfree_norm.pkl"),
    ("v11 seed 0", "touch/probs.pkl"), ("v11 seed 1", "touch/probs_seed1.pkl"), ("v11 seed 2", "touch/probs_seed2.pkl"),
    ("touch v2", "touch/probs2.pkl"), ("touch DFL-V4", "touch/probs_dflv4.pkl"),
    ("DFL trk-trained", "trainvar/probs_trk.pkl"), ("ball-smoothed train", "trainvar/probs_bsmooth.pkl"),
    ("Metrica keep 0.5", "v10var/probs_soft0_keep0.5_lgb.pkl"), ("Metrica keep 2", "v10var/probs_soft0_keep2_lgb.pkl"),
    ("soft targets 12", "v10var/probs_soft12_keep1_lgb.pkl"), ("extra trees", "v10var/probs_soft0_keep1_extra_treesTrue.pkl"),
    ("monotone", "v10var/probs_soft0_keep1_lgb_mono.pkl"), ("controller feats", "ctrlv10/probs.pkl"),
    ("ball height", "height/probs.pkl"), ("duel feats", "contest/probs_duel_seed0.pkl"),
    ("contest x3 weight", "contest/probs_w2_seed0.pkl"), ("self-training", "selftrain/probs_a1_c0.9.pkl"),
    ("aug", "aug/probs_aug.pkl"), ("bfree2", "bfree2/probs.pkl"), ("bfree3", "bfree3/probs.pkl"),
    ("a23", "a23/probs_ws0.3.pkl"), ("a23b", "a23b/probs_ws0.3.pkl"), ("tta", "tta/probs.pkl"),
]


def main():
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    _, pdl = exp_oracle.probs(SRC)
    rows = []
    for name, rel in CANDS:
        path = os.path.join(C, rel)
        if not os.path.isfile(path):
            continue
        d = bench._load(path)
        if SRC not in d or d[SRC] is None or any(x is None for x in d[SRC]):
            continue
        ph = exp_smooth.smooth(d[SRC], 25)
        cells = {}
        for conv in ("evctrl", "trk"):
            labs = bench.labels("dfl", conv)
            for dname, pds in (("L", pdl), ("T", [np.where(g == 2, 1 - 1e-4, 1e-4) for g in labs])):
                cells[conv + dname] = e2.score(SRC, conv, e2.decode_all(DEC, pds, ph)).mean()
        rows.append((name, cells))
        print("%-22s evctrl %.4f / %.4f | trk %.4f / %.4f" % (name, cells["evctrlL"], cells["evctrlT"], cells["trkL"],
                                                               cells["trkT"]), flush=True)
    ref = np.mean([[r[1]["evctrlL"], r[1]["evctrlT"]] for r in rows if r[0].startswith("v10 seed")], 0)
    print("v10 mean of 3 seeds on evctrl: %.4f / %.4f; candidates sorted by evctrl mean gain over it:" % tuple(ref))
    for name, c in sorted(rows, key=lambda r: -(r[1]["evctrlL"] + r[1]["evctrlT"])):
        print("  %-22s %+.4f / %+.4f" % (name, c["evctrlL"] - ref[0], c["evctrlT"] - ref[1]))


if __name__ == "__main__":
    main()
