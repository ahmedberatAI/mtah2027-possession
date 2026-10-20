"""Scoring helpers: the official metric on the public window, plus label-free full-match
diagnostics so that model choices are not made on 17 possessions alone."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFFICIAL = os.path.join(ROOT, "external", "more-than-a-hack-2027")
sys.path.insert(0, os.path.join(OFFICIAL, "evaluation"))
import possession_metrics as pm  # noqa: E402

import atd  # noqa: E402

GT_PUBLIC = os.path.join(OFFICIAL, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
LETTERS = np.array(["H", "A", "D", "X"])
_gt_cache = {}


def public_gt():
    if "gt" not in _gt_cache:
        _gt_cache["gt"] = pm.label_codes(pm.read_label_csv(GT_PUBLIC))
    return _gt_cache["gt"]


def to_codes(labels):
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        return labels.astype(np.int8)
    return pm.label_codes(labels)


def score(labels, bootstrap=0):
    """Official metrics on the public 5-minute window."""
    r = pm.evaluate(public_gt(), to_codes(labels), bootstrap_n=bootstrap)
    return {"macroF1": r["macro_f1"], "spellF1": r["spell_f1_mean"], "n_pred": r["n_pred_spells"],
            "n_gt": r["n_gt_spells"], "dur": r["mean_spell_duration_s"],
            "alive_dead_f1": r["alive_dead_f1"], "team_acc": r["team_acc_both_alive"],
            "spell_lo": r["spell_bootstrap_low"], "spell_hi": r["spell_bootstrap_high"],
            "f1_per_iou": {k: round(v["f1"], 3) for k, v in r["spell_f1"].items()}}


def diagnostics(labels):
    """What the submission claims about the whole match (no ground truth involved)."""
    codes = to_codes(labels)
    graded = atd.graded_mask(len(codes))
    runs = [(a, b + 1) for _, a, b in atd.HALVES]
    spells = pm.possession_spells(codes, runs)
    segs = pm.segments(codes, runs)
    lens = np.array([b - a for a, b, _ in spells]) / atd.FPS
    dsegs = [(a, b) for a, b, k in segs if k == pm.D]
    dl = np.array([b - a for a, b in dsegs]) / atd.FPS
    g = codes[graded]
    return {
        "n_poss": len(spells),
        "poss_mean_s": float(lens.mean()) if len(lens) else 0.0,
        "poss_median_s": float(np.median(lens)) if len(lens) else 0.0,
        "poss_lt2s": int((lens < 2).sum()),
        "n_dead_segs": len(dsegs),
        "dead_median_s": float(np.median(dl)) if len(dl) else 0.0,
        "dead_share": float((g == pm.D).mean()),
        "home_share": float((g == pm.H).sum() / max(1, ((g == pm.H) | (g == pm.A)).sum())),
        "X_graded": int((g == pm.X).sum()),
    }


def write(labels, path):
    codes = to_codes(labels)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(codes))
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, atd.N_FRAMES["tactical"])
    if reason:
        raise ValueError("invalid submission {}: {}".format(path, reason))
    return path


def fmt(d):
    return "  ".join("{}={}".format(k, ("%.4f" % v) if isinstance(v, float) else v) for k, v in d.items())
