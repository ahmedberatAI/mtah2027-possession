"""Round 11, idea B1 (docs/RAPOR.md): player names in the broadcast commentary as team-possession evidence.

The words come from colab/asr_whisper.py (faster-whisper on the broadcast audio, word timestamps), stored locally in
cache/asr/asr_words.json.gz (frame-level derivative of the match audio: never printed, never published). Names are
spotted with src/squads.py (public 2024-25 squads). AGGREGATE output only.

Broadcast audio time -> tactical frame: the broadcast is a real-time recording cut at halftime (src/broadcast.py):
tactical = broadcast + 4014 up to broadcast frame 86637 (end of the 1st half), + 26920 afterwards.

    python -I src/commentary.py diag
"""
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atd  # noqa: E402
import evaluate as ev  # noqa: E402
import squads  # noqa: E402

ROOT = os.path.dirname(HERE)
WORDS = os.path.join(ROOT, "cache", "asr", "asr_words.json.gz")
FPS = 25
CUT_B = 86637                    # last broadcast frame of the 1st half
N = atd.N_FRAMES["tactical"]


def to_tactical(t_sec):
    b = np.round(np.asarray(t_sec, float) * FPS).astype(np.int64)
    return np.where(b <= CUT_B, b + 4014, b + 26920), b


def mentions(kind="player"):
    """(tactical frame of the word start, team 0 home / 1 away, broadcast frame, canonical name) per mention."""
    with gzip.open(WORDS, "rt", encoding="utf-8") as fh:
        words = json.load(fh)
    ms = [m for m in squads.spot([(w[0], w[1], w[2]) for w in words]) if kind is None or m[3] == kind]
    f, b = to_tactical([m[0] for m in ms])
    return f, np.array([m[2] for m in ms], np.int8), b, [m[4] for m in ms], len(words)


def v10_state():
    codes = ev.pm.label_codes(ev.pm.read_label_csv(os.path.join(ROOT, "submissions", "candidate_v10.csv")))
    import pickle
    with open(os.path.join(ROOT, "cache", "v10", "barath_probs.pkl"), "rb") as fh:
        probs = pickle.load(fh)
    ph = np.full(N, np.nan)
    for a, (_, p) in probs.items():
        ph[a:a + len(p)] = p
    return np.asarray(codes), ph


def broadcast_valid():
    import pandas as pd
    hom = pd.read_csv(atd.paths("broadcast")["homography"], usecols=["valid"])
    return hom["valid"].values.astype(bool)


def spells(codes):
    """Maximal runs of H (0) or A (1) inside the graded halves: list of (start, end_exclusive, team)."""
    out = []
    for _, a, b in atd.HALVES:
        c = codes[a:b + 1]
        e = np.flatnonzero(np.diff(c) != 0) + 1
        starts = np.concatenate(([0], e))
        ends = np.concatenate((e, [len(c)]))
        for s, t in zip(starts, ends):
            if c[s] in (ev.pm.H, ev.pm.A):
                out.append((a + s, a + t, int(c[s] == ev.pm.A)))
    return out


def diag():
    f, team, b, names, n_words = mentions("player")
    ft, teamt, bt, _, _ = mentions("team")
    codes, ph = v10_state()
    valid_b = broadcast_valid()
    graded = atd.graded_mask(N)
    ok = (f >= 0) & (f < N)
    f, team, b = f[ok], team[ok], b[ok]
    vb = valid_b[np.clip(b, 0, len(valid_b) - 1)]
    alive = np.isin(codes, (ev.pm.H, ev.pm.A)) & graded
    live_min = alive.sum() / FPS / 60
    in_g = graded[f]
    print("words %d | player mentions %d (graded %d; home %d / away %d), team-name mentions %d" % (
        n_words, len(f), in_g.sum(), (team[in_g] == 0).sum(), (team[in_g] == 1).sum(), len(ft)))
    at_alive = alive[f]
    print("player mentions while v10 says alive: %d = %.2f / live min; with a valid broadcast view %.2f; "
          "v10 home share of alive frames %.2f, of mentions %.2f" % (
              at_alive.sum(), at_alive.sum() / live_min, vb[at_alive].mean(),
              (codes[alive] == ev.pm.H).mean(), (team[at_alive] == 0).mean()))
    # confident reference: inside v10 spells of >= 6 s, at least 1 s from either end, |p - 0.5| > 0.4
    conf = np.zeros(N, bool)
    team_at = np.full(N, -1, np.int8)
    for s, e, k in spells(codes):
        team_at[s:e] = k
        if e - s >= 6 * FPS:
            conf[s + FPS:e - FPS] = True
    conf &= np.abs(np.nan_to_num(ph, nan=0.5) - 0.5) > 0.4
    print("agreement of the named player's team with v10 at (mention - lag); all alive / confident frames only:")
    for sel_name, sel in (("valid view", vb), ("all", np.ones(len(f), bool))):
        row = []
        for lag in (-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6):
            g = f - int(round(lag * FPS))
            okg = (g >= 0) & (g < N) & sel
            g, tm = g[okg], team[okg]
            a1 = team_at[g] >= 0
            a2 = conf[g]
            row.append("%+.1fs %.3f/%.3f (n %d)" % (lag, (team_at[g][a1] == tm[a1]).mean(),
                                                   (team_at[g][a2] == tm[a2]).mean(), a2.sum()))
        print("  %-10s " % sel_name + " | ".join(row))
    # v10's live changes: is the NEW team named shortly after?
    sp = spells(codes)
    rows = {"new": 0, "old": 0, "n": 0}
    base = []
    for (s0, e0, k0), (s1, e1, k1) in zip(sp[:-1], sp[1:]):
        if e0 != s1 or k0 == k1 or e0 - s0 < FPS or e1 - s1 < FPS:
            continue
        c = s1
        w = (f >= c - FPS) & (f < c + 3 * FPS) & vb
        rows["n"] += 1
        rows["new"] += bool((team[w] == k1).any())
        rows["old"] += bool((team[w] == k0).any())
    # baseline: random 4 s windows inside confident frames
    rng = np.random.default_rng(0)
    cand = np.flatnonzero(conf)
    for c in rng.choice(cand, 2000, replace=False):
        w = (f >= c - FPS) & (f < c + 3 * FPS) & vb
        base.append(w.any())
    print("v10 live changes (both spells >= 1 s, valid view): %d; a mention of the NEW team in [-1, +3] s: %.2f, "
          "of the OLD team: %.2f; any mention in a random confident 4 s window: %.2f" % (
              rows["n"], rows["new"] / max(1, rows["n"]), rows["old"] / max(1, rows["n"]), np.mean(base)))
    # contrary mentions at the best typical lag (1 s): candidates for missed short spells
    g = np.clip(f - FPS, 0, N - 1)
    contra = (team_at[g] >= 0) & (team_at[g] != team) & vb
    p_t = np.where(team == 0, ph[g], 1 - ph[g])
    win_max = np.array([np.nanmax(np.where(team[i] == 0, ph[max(0, g[i] - 2 * FPS):g[i] + 2 * FPS],
                                           1 - ph[max(0, g[i] - 2 * FPS):g[i] + 2 * FPS]))
                        if contra[i] else np.nan for i in range(len(f))])
    print("contrary mentions (named team != v10 at -1 s, valid view): %d = %.2f / live min; v10 raw prob of the named "
          "team > 0.3 within +-2 s: %.2f; > 0.5: %.2f; median p at -1 s %.2f" % (
              contra.sum(), contra.sum() / live_min, np.mean(win_max[contra] > 0.3), np.mean(win_max[contra] > 0.5),
              np.nanmedian(p_t[contra])))


def evidence(lag_s=1.0, half_w_s=1.0, beta=1.0, cap=3.0, valid_only=True):
    """Per tactical frame: + for home, - for away. Each player mention at word start f adds beta to the team's
    log-odds on [f - lag - half_w, f - lag + half_w]; overlapping mentions add up, clipped at +-cap."""
    f, team, b, _, _ = mentions("player")
    keep = (f >= 0) & (f < N)
    if valid_only:
        vb = broadcast_valid()
        keep &= vb[np.clip(b, 0, len(vb) - 1)]
    f, team = f[keep], team[keep]
    c = f - int(round(lag_s * FPS))
    hw = int(round(half_w_s * FPS))
    acc = np.zeros(N + 1)
    sgn = np.where(team == 0, 1.0, -1.0)
    np.add.at(acc, np.clip(c - hw, 0, N), sgn)
    np.add.at(acc, np.clip(c + hw + 1, 0, N), -sgn)
    return np.clip(beta * np.cumsum(acc)[:N], -cap, cap)


def fused_probs(ev_llr, base_path=None, tag="comm"):
    import pickle
    base_path = base_path or os.path.join(ROOT, "cache", "v10", "barath_probs.pkl")
    with open(base_path, "rb") as fh:
        probs = pickle.load(fh)
    out = {}
    for a, (pd_, p) in probs.items():
        p = np.clip(p, 1e-4, 1 - 1e-4)
        z = np.log(p / (1 - p)) + ev_llr[a:a + len(p)]
        out[a] = (pd_, 1 / (1 + np.exp(-z)))
    path = os.path.join(ROOT, "cache", "camera", "probs_%s.pkl" % tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return path


def cmd_run():
    import camera
    base = camera.decode(os.path.join(ROOT, "cache", "v10", "barath_probs.pkl"))
    camera.describe("v10", base)
    for lag, hw, beta, cap in ((1.0, 1.0, 1.0, 2.0), (1.0, 1.0, 2.0, 3.0), (0.5, 1.5, 1.0, 2.0), (1.5, 1.0, 1.0, 2.0)):
        lab = camera.decode(fused_probs(evidence(lag, hw, beta, cap), tag="comm_l%g_h%g_b%g_c%g" % (lag, hw, beta, cap)))
        camera.describe("comm lag %g hw %g b %g c %g" % (lag, hw, beta, cap), lab, base)
        sys.stdout.flush()


if __name__ == "__main__":
    {"diag": diag, "run": cmd_run}[sys.argv[1] if len(sys.argv) > 1 else "diag"]()
