"""Ball-missing profile of labelled play: missing rates in alive / dead frames, and how missing runs
that start in a stoppage carry on after the restart. Compares BAR-ATH (aggregate output only) with
the degraded DFL / Metrica data the team model is trained and calibrated on (sim.degrade).

  .venv/Scripts/python.exe -I src/ball_gaps.py barath   # BAR-ATH: public GT window + model labels
  .venv/Scripts/python.exe -I src/ball_gaps.py sim      # degraded DFL / Metrica caches

Definitions (per half, then summed):
  alive / dead missing   share of alive / dead frames without a ball position
  spill                  alive frames covered by missing runs that started in a dead frame
  restart carry          frames from a restart (first alive frame after D) until the ball is seen,
                         capped at the next D start; 0 when the ball is seen at the restart
"""
import os
import pickle
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
FPS = 25


def runs_of(mask):
    m = np.concatenate(([0], np.asarray(mask, np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def profile(lab, seen):
    """Counts for one continuous block (a half). lab: 2 = dead, anything else alive."""
    lab = np.asarray(lab)
    seen = np.asarray(seen, bool)
    n = len(lab)
    dead = lab == 2
    alive = ~dead
    miss = ~seen
    cs_alive = np.concatenate(([0], np.cumsum(alive)))
    seg_end = np.full(n, n)
    dsegs = runs_of(dead)
    for a, b in dsegs:
        seg_end[a:b] = b
    next_dead = np.full(n + 1, n)
    for a, _ in reversed(dsegs):
        next_dead[:a + 1] = np.minimum(next_dead[:a + 1], a)
    out = {"n_alive": int(alive.sum()), "n_dead": int(dead.sum()),
           "miss_alive": int((miss & alive).sum()), "miss_dead": int((miss & dead).sum()),
           "spill": 0, "alive_started_in_alive": 0, "alive_runs": [], "alive_runs_into_dead": 0,
           "dead_runs": 0, "dead_runs_with_restart": 0, "dead_runs_past_restart": 0,
           "carry": [], "carry_spanning": 0, "entries": 0, "entries_missing": 0,
           "dead_segs": len(dsegs), "seen_runs": [b - a for a, b in runs_of(seen)],
           "miss_runs": []}
    for a, b in runs_of(miss):
        al = int(cs_alive[b] - cs_alive[a])
        out["miss_runs"].append(b - a)
        if dead[a]:
            out["dead_runs"] += 1
            out["spill"] += al
            if seg_end[a] < n:
                out["dead_runs_with_restart"] += 1
                out["dead_runs_past_restart"] += int(b > seg_end[a])
        else:
            out["alive_runs"].append(b - a)
            out["alive_started_in_alive"] += al
            out["alive_runs_into_dead"] += int(dead[a:b].any())
    miss_start = np.full(n, -1)
    miss_end = np.full(n, -1)
    for a, b in runs_of(miss):
        miss_start[a:b] = a
        miss_end[a:b] = b
    out["lead"] = []
    out["dead_runs_per_seg"] = []
    prev_b = 0
    for a, b in dsegs:
        out["dead_runs_per_seg"].append(int(np.sum(miss[a:b] & ~np.concatenate(([False], miss[a:b - 1])))))
        if a > 0:
            out["entries"] += 1
            out["entries_missing"] += int(miss[a - 1])
            if miss[a - 1]:
                # how long before the D start the ball went missing (capped at the previous restart)
                out["lead"].append(int(a - max(miss_start[a - 1], prev_b)))
        prev_b = b
        if b >= n:
            continue
        if seen[b]:
            out["carry"].append(0)
            continue
        out["carry"].append(int(min(miss_end[b], next_dead[b]) - b))
        out["carry_spanning"] += int(miss_start[b] < a)
    return out


def merge(profiles):
    tot = {}
    for p in profiles:
        for k, v in p.items():
            tot[k] = tot.get(k, [] if isinstance(v, list) else 0) + v
    return tot


def summary(p):
    """Rates and distributions from merged counts (aggregates only)."""
    c = np.array(p["carry"], float) / FPS
    ar = np.array(p["alive_runs"], float) / FPS
    q = lambda x, v: float(np.percentile(x, v)) if len(x) else float("nan")
    return {
        "alive_min": p["n_alive"] / FPS / 60, "dead_share": p["n_dead"] / max(1, p["n_alive"] + p["n_dead"]),
        "alive_missing": p["miss_alive"] / max(1, p["n_alive"]),
        "dead_missing": p["miss_dead"] / max(1, p["n_dead"]),
        "spill_share_alive": p["spill"] / max(1, p["n_alive"]),
        "alive_started_share_alive": p["alive_started_in_alive"] / max(1, p["n_alive"]),
        "alive_runs_per_alive_min": len(ar) / max(1e-9, p["n_alive"] / FPS / 60),
        "alive_run_median_s": q(ar, 50), "alive_run_p75_s": q(ar, 75),
        "alive_runs_into_dead": p["alive_runs_into_dead"] / max(1, len(ar)),
        "dead_runs_past_restart": p["dead_runs_past_restart"] / max(1, p["dead_runs_with_restart"]),
        "n_restarts": len(c), "restart_missing": float(np.mean(c > 0)) if len(c) else float("nan"),
        "carry_gt1s": float(np.mean(c > 1)) if len(c) else float("nan"),
        "carry_gt3s": float(np.mean(c > 3)) if len(c) else float("nan"),
        "carry_gt10s": float(np.mean(c > 10)) if len(c) else float("nan"),
        "carry_median_s_if_missing": q(c[c > 0], 50), "carry_p75_s_if_missing": q(c[c > 0], 75),
        "carry_p90_s_if_missing": q(c[c > 0], 90), "carry_mean_s": float(c.mean()) if len(c) else float("nan"),
        "carry_spanning": p["carry_spanning"] / max(1, int(np.sum(c > 0))),
        "entry_missing": p["entries_missing"] / max(1, p["entries"]),
        "lead_median_s": q(np.array(p["lead"]) / FPS, 50), "lead_p90_s": q(np.array(p["lead"]) / FPS, 90),
        "dead_runs_per_seg": float(np.mean(p["dead_runs_per_seg"])) if p["dead_runs_per_seg"] else float("nan"),
        "miss_run_median_s": q(np.array(p["miss_runs"]) / FPS, 50),
        "seen_run_median_s": q(np.array(p["seen_runs"]) / FPS, 50),
    }


COLS = [("alive_missing", "alive miss", "%.3f"), ("dead_missing", "dead miss", "%.3f"),
        ("spill_share_alive", "spill/alive", "%.3f"), ("alive_started_share_alive", "a-start/alive", "%.3f"),
        ("dead_share", "D share", "%.3f"), ("n_restarts", "restarts", "%d"),
        ("restart_missing", "miss@restart", "%.3f"), ("carry_gt1s", ">1s", "%.3f"), ("carry_gt3s", ">3s", "%.3f"),
        ("carry_gt10s", ">10s", "%.3f"), ("carry_median_s_if_missing", "carry med s", "%.1f"),
        ("carry_p90_s_if_missing", "carry p90 s", "%.1f"), ("carry_mean_s", "carry mean s", "%.2f"),
        ("dead_runs_past_restart", "d-runs past", "%.3f"), ("entry_missing", "miss@D-entry", "%.3f"),
        ("lead_median_s", "lead med s", "%.1f"), ("lead_p90_s", "lead p90 s", "%.1f"),
        ("dead_runs_per_seg", "runs/D seg", "%.2f"),
        ("alive_runs_per_alive_min", "a-runs/min", "%.2f"), ("alive_run_median_s", "a-run med s", "%.1f"),
        ("alive_runs_into_dead", "a-runs->D", "%.3f"), ("miss_run_median_s", "miss run med", "%.1f"),
        ("seen_run_median_s", "seen run med", "%.1f")]


def header():
    return "| set | " + " | ".join(c[1] for c in COLS) + " |\n|---|" + "---|" * len(COLS)


def row(name, s):
    return "| %s | %s |" % (name, " | ".join(fmt % s[k] for k, _, fmt in COLS))


# -- BAR-ATH (aggregates only) ----------------------------------------------------------------------

def barath_labelings(f):
    """Label arrays over the whole match (2 = D, 0/1 alive, 3 = not graded), with their blocks."""
    import atd
    import evaluate as ev
    import model_v3
    import model_v5
    halves = [(a, b + 1) for _, a, b in atd.HALVES]
    # from the 2nd-half kick-off on (the graded pre-kick-off stretch is D by rule, not a stoppage)
    play = [(halves[0][0], halves[0][1]), (atd.KICKOFF_2ND, halves[1][1])]
    gt = ev.public_gt()
    out = {"public GT window (5 min)": (gt, ev.pm.valid_runs(gt))}
    out["v5 pen_team 8 (current)"] = (model_v5.predict(f), play)
    out["v5 pen_team 20"] = (model_v5.predict(f, {"pen_team": 20.0}), play)
    blind = dict(w_missing=0.0, w_still=0.0, w_ballmove=0.0)
    out["v5, ball-presence cues off"] = (model_v5.predict(f, blind), play)
    out["v3 alive/dead, ball cues off"] = (model_v3.predict(f, dict(blind, w_team=0.0)), play)
    out["v3 alive/dead, ball + out cues off"] = (model_v3.predict(f, dict(blind, w_team=0.0, w_out=0.0)), play)
    return out


def barath():
    from make_submission import merged_features
    import evaluate as ev
    f = merged_features()
    labs = barath_labelings(f)
    lines = []
    for src, seen in (("merged ball (model input)", f["seen"]), ("tactical ball", f["tac_seen"])):
        lines += ["", "BAR-ATH, %s" % src, header()]
        for name, (lab, blocks) in labs.items():
            p = merge([profile(lab[a:b], seen[a:b]) for a, b in blocks])
            lines.append(row(name, summary(p)))
    # agreement of the model labellings with the public GT on alive/dead inside the window
    gt = ev.public_gt()
    g = gt < 3
    for name, (lab, _) in labs.items():
        if name.startswith("public"):
            continue
        lines.append("alive/dead agreement with public GT in the window, %s: %.3f" % (
            name, np.mean((lab[g] == 2) == (gt[g] == 2))))
    print("\n".join(lines))
    return lines


# -- degraded training / calibration data -----------------------------------------------------------

def dfl_profile(cache_dir):
    import dfl
    ps = []
    for m in dfl.MATCHES:
        with open(os.path.join(cache_dir, "{}_feat.pkl".format(m)), "rb") as fh:
            d = pickle.load(fh)
        ps += [profile(b["lab"], b["seen"]) for b in d["blocks"]]
    return merge(ps)


def metrica_profile(cache_dir, seeds=(0, 1, 2)):
    ps = []
    for s in seeds:
        with open(os.path.join(cache_dir, "probs_seed%d.pkl" % s), "rb") as fh:
            res = pickle.load(fh)
        for halves in res.values():
            ps += [profile(lab, seen) for _, _, lab, seen in halves]
    return merge(ps)


def sim_report(sets):
    lines = [header()]
    for name, kind, path in sets:
        if not os.path.isdir(path):
            continue
        try:
            p = dfl_profile(path) if kind == "dfl" else metrica_profile(path)
        except (IOError, OSError):
            continue
        lines.append(row(name, summary(p)))
    print("\n".join(lines))
    return lines


DEFAULT_SETS = [("degraded DFL (cache/dfl, sim v1)", "dfl", os.path.join(ROOT, "cache", "dfl")),
                ("degraded Metrica (cache/metrica, sim v1)", "metrica", os.path.join(ROOT, "cache", "metrica"))]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "barath"
    if cmd == "barath":
        barath()
    elif cmd == "sim":
        sim_report(DEFAULT_SETS)
