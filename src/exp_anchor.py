"""Round 8 (research fleet a02 / a07 / a12 / a17): event-anchor oracle. How much spell F1 could team-attributed PASS
anchors -- what the video pilot (colab/bas_pilot.py) produces -- add to v7 through a capped fusion, and how good must
they be?

Anchors from TRUE events (an optimistic upper bound: these sets' labels are built from the same events):
  Metrica      PASS events (start frame = the kick; team = the passer's)
  SkillCorner  player possessions ending in a pass (kick at frame_end; the player's team)
Corruption towards a video detector: keep a share `recall`, flip the team with probability 1 - `acc`, jitter the
time (Gaussian sd `jit` frames), add `fp` false anchors per alive minute with a random team.
Fusion: z = logit(p_v7) + beta * clip(sum_a s_a * 1[t_a - W0 <= t <= t_a + W1], -1, 1) (s = +1 home, -1 away),
v7 decoder (B2, pen 12); learned and true D; paired spell F1 against v7. AGGREGATE output only.

  python -I src/exp_anchor.py
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import exp_ctrl2 as e2  # noqa: E402
import exp_oracle  # noqa: E402
import metrica_sample as ms  # noqa: E402
import skillcorner as scr  # noqa: E402

W0, W1 = 12, 3              # the passer's team controls the ball in the 0.5 s before the kick
EPS = 1e-4
SCEN = (("perfect", dict(recall=1.0, acc=1.0, jit=0.0, fp=0.0)),
        ("pilot-like", dict(recall=0.6, acc=0.94, jit=3.0, fp=1.0)),
        ("pilot, no false", dict(recall=0.6, acc=0.94, jit=3.0, fp=0.0)),
        ("worse", dict(recall=0.5, acc=0.85, jit=5.0, fp=2.0)))
BETAS = (1.0, 2.0, 4.0)
SETS = (("metrica@v3", "ctrl", (30.0, 8, 50)), ("sc@v3", "pp", (30.0, 8, 50)), ("sc@v3", "phase", (30.0, 8, 50)))


def metrica_anchors(b, contacts=False, typed=False):
    g, h = int(b["group"][1:]), int(b["key"][-1])
    ev = ms.events(g)
    f0 = int(ms.parse(g)["h%d_n0" % h][0])
    ev = ev[ev['period'] == h]
    if not contacts:
        p = ev[ev['type'] == 'PASS']
        return np.stack([p['start'].to_numpy() - f0, p['team'].to_numpy()], 1).astype(np.int64)
    # All observable event-contact PROXIES, not every dribble touch. Losing a
    # challenge / receiving a foul does not establish that the actor touched it.
    # BALL OUT is used at its initiating action, never at the line-crossing end.
    contact = ev['type'].isin({'PASS', 'CARRY', 'RECOVERY', 'SHOT', 'BALL LOST', 'BALL OUT', 'SET PIECE'})
    won = (ev['type'] == 'CHALLENGE') & ev['tokens'].map(lambda t: 'WON' in t and 'FAULT' not in t)
    p = ev[contact | won]
    p = p[~p['tokens'].map(lambda t: bool(t & {'RETAKEN', 'END HALF', 'REFEREE HIT'}))]
    kind = np.where(p['type'].isin({'RECOVERY', 'CARRY', 'SET PIECE'}), 1,
                    np.where(p['type'] == 'CHALLENGE', 0, -1))
    rows = [np.stack([p['start'].to_numpy() - f0, p['team'].to_numpy(), kind], 1)]
    # Successful pass reception is another contact, by the receiving teammate.
    p = ev[(ev['type'] == 'PASS') & ~ev['tokens'].map(lambda t: bool(t & {'OFFSIDE', 'INTERCEPTION'}))]
    rows.append(np.stack([p['end'].to_numpy() - f0, p['team'].to_numpy(), np.ones(len(p))], 1))
    out = deduplicate_contacts(np.concatenate(rows).astype(np.int64))
    return out if typed else out[:, :2]


def sc_anchors(b, contacts=False, typed=False):
    m, h = b["group"], int(b["key"][-1])
    d = scr.parse(m)
    f0 = int(d["h%d_f0" % h][0])
    meta = json.load(open(os.path.join(scr.DATA, m, "%s_match.json" % m), encoding="utf-8"))
    home = meta["home_team"]["id"]
    ev = pd.read_csv(os.path.join(scr.DATA, m, "%s_dynamic_events.csv" % m), low_memory=False,
                     usecols=["event_type", "period", "frame_start", "frame_end", "team_id", "start_type", "end_type"])
    if contacts:
        pp = ev[(ev['event_type'] == 'player_possession') & (ev['period'] == h)]
        # Receptions, interceptions and recoveries establish a start contact.
        # keep_possession is a continuity marker; pressure/engagement is not contact.
        start = pp[pp['start_type'].isin({'recovery'}) | pp['start_type'].str.endswith(('_reception', '_interception'), na=False)]
        end = pp[pp['end_type'].isin({'pass', 'shot', 'clearance'})]
        rows = []
        for p, col, kind in ((start, 'frame_start', 1), (end, 'frame_end', -1)):
            t = np.round((p[col].to_numpy() - f0) * 2.5).astype(np.int64)
            rows.append(np.stack([t, np.where(p['team_id'].to_numpy() == home, 0, 1), np.full(len(p), kind)], 1))
        out = deduplicate_contacts(np.concatenate(rows))
        return out if typed else out[:, :2]
    p = ev[(ev["event_type"] == "player_possession") & (ev["period"] == h) & (ev["end_type"] == "pass")]
    t = np.round((p["frame_end"].to_numpy() - f0) * 2.5).astype(np.int64)
    return np.stack([t, np.where(p["team_id"].to_numpy() == home, 0, 1)], 1)


def deduplicate_contacts(anc, gap=3):
    """Collapse same-team annotations within 120 ms; retain opposing contacts."""
    rows, last = [], {0: -100000, 1: -100000}
    for row in anc[np.argsort(anc[:, 0], kind='stable')]:
        t, k = row[:2]
        if t - last[int(k)] > gap:
            rows.append(row)
            last[int(k)] = t
    return np.asarray(rows, dtype=np.int64).reshape(-1, anc.shape[1])


def corrupt(anc, n, alive_frames, rng, recall, acc, jit, fp):
    keep = anc[rng.random(len(anc)) < recall].copy()
    flip = rng.random(len(keep)) > acc
    keep[flip, 1] = 1 - keep[flip, 1]
    if jit > 0:
        keep[:, 0] += np.round(rng.normal(0, jit, len(keep))).astype(np.int64)
    n_fp = rng.poisson(fp * alive_frames / 1500.0)
    false = np.stack([rng.integers(0, n, n_fp), rng.integers(0, 2, n_fp)], 1) if n_fp else np.zeros((0, 2), np.int64)
    out = np.concatenate([keep, false])
    return out[(out[:, 0] >= 0) & (out[:, 0] < n)]


def evidence(anc, n, before=W0, after=W1):
    s = np.zeros(n + 1)
    for t, k in anc:
        a, e = max(0, t - before), min(n, t + after + 1)
        s[a] += 1.0 if k == 0 else -1.0
        s[e] -= 1.0 if k == 0 else -1.0
    return np.clip(np.cumsum(s)[:n], -1, 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    cache = {}
    for src, conv, dec in SETS:
        bl = bench.blocks(src)
        if src not in cache:
            phl, pdl = exp_oracle.probs(src)
            anchors = [metrica_anchors(b) if src.startswith("metrica") else sc_anchors(b) for b in bl]
            cache[src] = (phl, pdl, anchors)
            n_alive = sum(int((b["lab"] < 2).sum()) for b in bl)
            print("%-10s anchors %d (%.1f per alive minute)" % (src, sum(len(a) for a in anchors),
                                                                 sum(len(a) for a in anchors) / (n_alive / 1500.0)),
                  flush=True)
        phl, pdl, anchors = cache[src]
        labs = bench.labels(src, conv)
        for dname, pds in (("learned D", pdl), ("true D", [np.where(g == 2, 1 - EPS, EPS) for g in labs])):
            ref = e2.score(src, conv, e2.decode_all(dec, pds, phl))
            cells = []
            for sname, sp in SCEN:
                rng = np.random.default_rng(0)
                evs = [evidence(corrupt(a, len(p), int((l < 2).sum()), rng, **sp), len(p))
                       for a, p, l in zip(anchors, phl, labs)]
                for beta in BETAS:
                    phs = [e2.sig(e2.logit(p) + beta * ev_) for p, ev_ in zip(phl, evs)]
                    v = e2.score(src, conv, e2.decode_all(dec, pds, phs))
                    cells.append("%s b%g %+.4f (%d/%d)" % (sname, beta, v.mean() - ref.mean(), int((v > ref).sum()),
                                                            len(v)))
            print("%-10s %-6s %-9s v7 %.4f | %s" % (src, conv, dname, ref.mean(), " | ".join(cells)), flush=True)


if __name__ == "__main__":
    main()
