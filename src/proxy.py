"""Label-free, full-match proxy checks for a prediction. Smart Tagging is automatic and noisy,
so these are relative signals for comparing two versions, never targets to tune hard against.

  tag_team_agree : over possession-phase tags (team in possession) and defending-phase tags
                   (other team in possession), share of predicted alive frames on the implied team
  setpiece_ok    : share of team-tagged set pieces (corner/free kick/goal kick) whose window holds
                   a predicted D -> <tag team> restart
  restart_any    : share of all set-piece tags whose window holds any predicted D -> alive restart
  ballout_cov    : share of clear ball-out episodes (ball > 1 m outside, >= 0.2 s) labelled D
"""
import numpy as np

import features as F

POSS_CODES = ("BUILD UP", "PROGRESSION", "LONG BALL", "ATTACKING TRANSITION")
DEF_CODES = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENSIVE TRANSITION")
_tags = {}


def tags():
    if "t" not in _tags:
        _tags["t"] = F.load_tags()
    return _tags["t"]


def proxies(labels, f, exclude=(11185, 18685)):
    lab = np.asarray(labels)
    T = tags()
    n = len(lab)
    agree = tot = 0
    for _, r in T.iterrows():
        if r.team < 0 or r.code not in POSS_CODES + DEF_CODES:
            continue
        a, b = max(0, r.sf), min(n, r.ef)
        if b > exclude[0] and a < exclude[1]:
            continue
        team = r.team if r.code in POSS_CODES else 1 - r.team
        seg = lab[a:b]
        alive = seg[(seg == 0) | (seg == 1)]
        tot += len(alive)
        agree += int((alive == team).sum())
    sp = T[T.code == "SET PIECES"]
    ok_team = n_team = ok_any = 0
    for _, r in sp.iterrows():
        a, b = max(0, r.sf - 50), min(n, r.ef)
        seg = lab[a:b]
        restarts = np.flatnonzero((seg[:-1] == 2) & (seg[1:] < 2)) + 1
        ok_any += int(len(restarts) > 0)
        if r.team >= 0:
            n_team += 1
            ok_team += int(any(seg[i] == r.team for i in restarts))
    out = np.nan_to_num(f["ball_out"], nan=-1) > 1.0
    eps = [(a, b) for a, b in F.runs_of(out & f["graded"]) if b - a >= 5]
    cov = np.mean([np.mean(lab[a:b] == 2) for a, b in eps]) if eps else float("nan")
    return {"tag_team_agree": agree / max(1, tot), "setpiece_ok": ok_team / max(1, n_team),
            "restart_any": ok_any / max(1, len(sp)), "ballout_cov": float(cov)}
