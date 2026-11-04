"""Round 35 (docs/RAPOR.md): same-day screen for short-possession tricks. A trick adds features to the verifier's candidate
rows of v15 (exp_kpu C1abETmS); the leave-one-match-out verifier is rerun with and without them on one v10 seed and the
difference is reported: insert AUC per source (the rows' own y, and the strict label IoU >= 0.5 with a true spell of the
team) and spell F1 per context with v15's thresholds. Screen: insert AUC +0.01 on DFL and Metrica, or spell F1 >= +0.003 on
both. AGGREGATE output only.

  python -I src/exp_trick.py screen NAME[,NAME...] [SEED]
  python -I src/exp_trick.py signal NAME[,NAME...] [SEED]  a small logistic correction instead of more trees, and feature AUCs
  python -I src/exp_trick.py arith                         group A: decision rules on the stored probabilities
  python -I src/exp_trick.py target strict,votes,rank [SEED]  group B: what the verifier is trained to predict
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import bench  # noqa: E402
import exp_kpu  # noqa: E402
import exp_refine as er  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_verify as ev  # noqa: E402
import metrica_sample as ms  # noqa: E402
import short_specialist as ss  # noqa: E402

BASE = "C1abETmS"
SCORE = {"dfl@v3": ("evctrl", "trk"), "metrica@v3": ("ctrl",), "sc@v3": ("phase", "pp")}
TRICKS = {}


def trick(fn):
    """Register fn(ctx, rows) -> feature names; it sets those keys on the rows of one half (both dead-ball channels)."""
    TRICKS[fn.__name__] = fn
    return fn


class Ctx:
    """One half of a source: the block and, computed on demand, the frame cues and every member's raw evidence."""
    _members = {}

    def __init__(self, src, i, block):
        self.src, self.i, self.b = src, i, block
        self._cues = None

    @property
    def cues(self):
        if self._cues is None:
            self._cues = ea.frame_cues(self.b)
        return self._cues

    @classmethod
    def members(cls, src):
        """{"v10": [3 seeds], "gru": [3 members], "c1": [3 seeds]}: per half P(home) of every member, held out as in v15."""
        if src not in cls._members:
            gru = ev.lomo_bigru() if src.startswith("dfl") else es.bigru_members(src)
            outs = exp_kpu.outputs("C1", src)
            cls._members = {src: dict(v10=[es.v10_raw(k, src) for k in es.SEEDS], gru=[gru[k] for k in es.SEEDS],
                                      c1=[[ss.pk(np.asarray(o, np.float32), 0) for o in outs[ss.SEEDS[k]]] for k in es.SEEDS])}
        return cls._members[src]


def _toward(p, k):
    """Logit of P(home) seen from team k."""
    lg = ea.logit(p)
    return lg if k == 0 else -lg


@trick
def spread(ctx, rows):
    """T4 seed: how much the members of each model family agree on team k inside the candidate: mean and spread of their
    mean margin, and the weakest member."""
    mem = Ctx.members(ctx.src)
    names = []
    for fam, runs in mem.items():
        names += ["tr_%s_std" % fam, "tr_%s_min" % fam]
        lg = [ea.logit(np.asarray(run[ctx.i], np.float64)) for run in runs]
        for r in rows:
            sign = 1.0 if r["k"] == 0 else -1.0
            vals = [sign * float(np.mean(x[r["s"]:r["t"]])) for x in lg]
            r["tr_%s_std" % fam], r["tr_%s_min" % fam] = float(np.std(vals)), float(np.min(vals))
    return names


# -- group C: what the 22 dots do around a candidate (T1); no stable identities: short slot velocities, jumps masked --------

class Motion:
    """Per half: positions and +-5 frame velocities in metres (m/s; > 12 m/s = a track swap, masked), attack signs, ball."""

    def __init__(self, ctx):
        c = ctx.cues
        self.c, self.b = c, ctx.b
        self.pm, self.team, self.gk = c["pxy"].astype(np.float64), np.asarray(c["team"]), np.asarray(c["gk"], bool)
        self.bxy = c["bxy"].astype(np.float64)
        self.n = len(self.pm)
        v = np.full_like(self.pm, np.nan)
        v[5:-5] = (self.pm[10:] - self.pm[:-10]) / 0.4
        v[np.linalg.norm(v, axis=2) > 12] = np.nan
        self.v = v
        self.sign = ss._signs(ctx.b["P"], self.team, self.gk)
        self.out = (self.team == 0) & ~self.gk, (self.team == 1) & ~self.gk
        self.dk = c["dk"]

    def clip(self, a, b):
        return max(0, int(a)), min(self.n, int(b))

    def anchor(self, u):
        """The ball when a player is within 3 m of it, else the midpoint of a clearly closest opposing pair (< 4 m), else None."""
        u = min(max(0, int(u)), self.n - 1)
        b = self.bxy[u]
        p = self.pm[u]
        if np.isfinite(b[0]):
            d = np.linalg.norm(p - b, axis=1)
            if np.isfinite(d).any() and np.nanmin(d) <= 3:
                return b
        x, y = p[self.team == 0], p[self.team == 1]
        x, y = x[np.isfinite(x[:, 0])], y[np.isfinite(y[:, 0])]
        if not len(x) or not len(y):
            return None
        d = np.linalg.norm(x[:, None] - y[None], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        dd = np.sort(d.ravel())
        if dd[0] < 4 and (len(dd) < 2 or dd[1] - dd[0] >= 1):
            return 0.5 * (x[i] + y[j])
        return None

    def reaction(self, q, f0, f1, k):
        """Team k's and the other team's local radial velocity (players 4-14 m from q, outward > 0) and the share of their far
        players (14-30 m) running towards their own goal, scaled to comparable units."""
        f0, f1 = self.clip(f0, f1)
        if q is None or f1 - f0 < 3:
            return None
        p, v = self.pm[f0:f1], self.v[f0:f1]
        rel = p - q
        d = np.linalg.norm(rel, axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            rad = np.sum(v * rel, axis=2) / d
        out = []
        for t in (1 - k, k):
            m = self.out[t]
            near = (d[:, m] >= 4) & (d[:, m] <= 14) & np.isfinite(rad[:, m])
            far = (d[:, m] >= 14) & (d[:, m] <= 30) & np.isfinite(v[:, m, 0])
            back = -self.sign[t] * v[:, m, 0] > 2
            out.append((np.median(rad[:, m][near]) / 2.0 if near.any() else np.nan,
                        np.mean(back[far]) / 0.3 if far.any() else np.nan))
        return np.array([out[0][0], out[1][0], out[0][1], out[1][1]])


def _motion(ctx):
    if not hasattr(ctx, "_m"):
        ctx._m = Motion(ctx)
    return ctx._m


def _pair(m, s, t, k):
    """T1 #1: reaction changes across s and t; Q > 0 when the second reverses the first."""
    d = (t - s) / ea.FPS
    h = min(0.8, (d - 0.4) / 2)
    if h < 0.24:
        return None
    fh = int(round(h * ea.FPS))
    qs, qt = m.anchor(s), m.anchor(t)
    a0, a1 = m.reaction(qs, s - 5 - fh, s - 5, k), m.reaction(qs, s + 5, s + 5 + fh, k)
    b0, b1 = m.reaction(qt, t - 5 - fh, t - 5, k), m.reaction(qt, t + 5, t + 30, k)
    b2 = m.reaction(qt, t + 30, t + 60, k)
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return None
    ds, dt = np.nan_to_num(a1 - a0), np.nan_to_num(b1 - b0)
    q1 = -float(ds @ dt) / (np.linalg.norm(ds) * np.linalg.norm(dt) + 0.1)
    e1 = float(min(np.linalg.norm(ds), np.linalg.norm(dt)))
    if b2 is None:
        q2, e2 = np.nan, np.nan
    else:
        d2 = np.nan_to_num(b2 - b0)
        q2 = -float(ds @ d2) / (np.linalg.norm(ds) * np.linalg.norm(d2) + 0.1)
        e2 = float(min(np.linalg.norm(ds), np.linalg.norm(d2)))
    return ds, dt, q1, e1, q2, e2


REV = ["tr_rev_ds%d" % j for j in range(4)] + ["tr_rev_dt%d" % j for j in range(4)] + ["tr_rev_q", "tr_rev_e", "tr_rev_q2", "tr_rev_e2"]
PLA = ["tr_rev_qrel", "tr_rev_erel", "tr_rev_qrank", "tr_rev_qshift"]


@trick
def reversal(ctx, rows):
    """T1 #1 + #4: two reversals outside the scrum, and the same score at placebo times inside the same decoded possession."""
    m = _motion(ctx)
    for r in rows:
        s, t, k = r["s"], r["t"], r["k"]
        got = _pair(m, s, t, k)
        for key in REV + PLA:
            r[key] = np.nan
        if got is None:
            continue
        ds, dt, q1, e1, q2, e2 = got
        for j in range(4):
            r["tr_rev_ds%d" % j], r["tr_rev_dt%d" % j] = float(ds[j]), float(dt[j])
        r["tr_rev_q"], r["tr_rev_e"], r["tr_rev_q2"], r["tr_rev_e2"] = q1, e1, q2, e2
        qs, es = [], []
        for delta in (-60, -30, 30, 60):
            if r["a"] <= s + delta and t + delta <= r["e"]:
                other = _pair(m, s + delta, t + delta, k)
                if other is not None:
                    qs.append(other[2])
                    es.append(other[3])
        if len(qs) >= 2:
            r["tr_rev_qrel"], r["tr_rev_erel"] = q1 - float(np.median(qs)), e1 - float(np.median(es))
            r["tr_rev_qrank"], r["tr_rev_qshift"] = float(np.mean(np.array(qs) < q1)), float(np.max(qs))
    return REV + PLA


def _ray(bxy, f0, f1):
    """Least-squares straight line through the finite ball samples: point, unit direction (of travel)."""
    f0, f1 = max(f0, 0), min(f1, len(bxy))
    if f1 - f0 < 3:
        return None
    y = bxy[f0:f1]
    ok = np.isfinite(y[:, 0])
    if ok.sum() < 3:
        return None
    tt = np.arange(f0, f1)[ok].astype(float)
    a = np.polyfit(tt, y[ok, 0], 1)
    b = np.polyfit(tt, y[ok, 1], 1)
    v = np.array([a[0], b[0]])
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return np.array([np.polyval(a, tt[-1]), np.polyval(b, tt[-1])]), v / n, n * ea.FPS


def _off_ray(ray, q):
    d = q - ray[0]
    along = float(d @ ray[1])
    return float(np.linalg.norm(d - along * ray[1])), along


def _isolated(m, f0, f1, k, near=2.0, far=3.0, hold=3):
    """First frame in [f0, f1) that opens >= hold frames with team k within `near` of the seen ball and the other beyond `far`."""
    f0, f1 = m.clip(f0, f1)
    ok = (m.dk[k][f0:f1] < near) & (m.dk[1 - k][f0:f1] > far) & np.isfinite(m.bxy[f0:f1, 0])
    run = 0
    for j, x in enumerate(ok):
        run = run + 1 if x else 0
        if run >= hold:
            return f0 + j - hold + 1
    return None


PLAN = ["tr_plan_res", "tr_plan_cos", "tr_plan_miss", "tr_plan_on", "tr_plan_flag", "tr_plan_gap"]


@trick
def plan(ctx, rows):
    """T1 #2: did the next receiver of the old team simply continue the run he had before the candidate? A brushed pass leaves the
    intended receiver's run intact; real control by the other team disrupts it."""
    m = _motion(ctx)
    for r in rows:
        s, t, k = r["s"], r["t"], r["k"]
        for key in PLAN:
            r[key] = np.nan
        old = 1 - k
        u = _isolated(m, t, t + 30, old)
        f = s - 10
        if u is None or f < 0:
            continue
        ball_u = m.bxy[u]
        p_u = m.pm[u]
        d = np.linalg.norm(p_u - ball_u, axis=1)
        d[m.team != old] = np.inf
        d[~np.isfinite(d)] = np.inf
        rec = int(np.argmin(d))
        r["tr_plan_gap"] = (u - t) / ea.FPS
        idx = np.flatnonzero(m.out[old] & np.isfinite(m.pm[f, :, 0]) & np.isfinite(m.v[f, :, 0]))
        if not len(idx):
            continue
        hat = m.pm[f, idx] + (u - f) / ea.FPS * m.v[f, idx]
        res = np.linalg.norm(hat - ball_u, axis=1)
        j = idx[int(np.argmin(res))]
        r["tr_plan_res"] = float(res.min())
        v0, v1 = m.v[f, j], m.v[u, rec] if u < m.n else np.array([np.nan, np.nan])
        if np.isfinite(v0).all() and np.isfinite(v1).all() and np.linalg.norm(v0) > 0.5 and np.linalg.norm(v1) > 0.5:
            r["tr_plan_cos"] = float(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        ray = _ray(m.bxy, s - 20, s - 4)
        if ray is not None:
            r["tr_plan_miss"] = _off_ray(ray, ball_u)[0]
            b_s = m.bxy[s]
            if np.isfinite(b_s[0]):
                r["tr_plan_on"] = _off_ray(ray, b_s)[0]
        vals = [r["tr_plan_res"], r["tr_plan_cos"], r["tr_plan_miss"], r["tr_plan_on"]]
        if np.isfinite(vals).all():
            r["tr_plan_flag"] = float(vals[0] < 2.5 and vals[1] > 0.85 and vals[2] < 2 and vals[3] < 2)
    return PLAN


def _lane(q, tgt, opp):
    """Smallest distance of the other team's players to the middle 70% of the segment q -> tgt."""
    seg = tgt - q
    n2 = float(seg @ seg)
    if n2 < 1e-6 or not len(opp):
        return np.nan
    w = np.clip((opp - q) @ seg / n2, 0.15, 0.85)
    return float(np.min(np.linalg.norm(opp - (q + w[:, None] * seg), axis=1)))


def _outlets(m, q, f0, f1, k):
    """Team k's new passing outlets over [f0, f1]: players 5-22 m from q, never within 4 m of it, moving > 1 m away from it and
    opening their lane by > 0.5 m to a clearance > 1.5 m."""
    f0, f1 = m.clip(f0, f1)
    if f1 - f0 < 5:
        return []
    p0, p1 = m.pm[f0], m.pm[f1 - 1]
    out = []
    opp0, opp1 = p0[m.team == 1 - k], p1[m.team == 1 - k]
    opp0, opp1 = opp0[np.isfinite(opp0[:, 0])], opp1[np.isfinite(opp1[:, 0])]
    for j in np.flatnonzero(m.out[k]):
        a, b = p0[j], p1[j]
        if not (np.isfinite(a[0]) and np.isfinite(b[0])):
            continue
        d0, d1 = np.linalg.norm(a - q), np.linalg.norm(b - q)
        if not 5 <= d0 <= 22 or d1 < 4:
            continue
        l0, l1 = _lane(q, a, opp0), _lane(q, b, opp1)
        if d1 - d0 > 1 and np.isfinite(l0) and np.isfinite(l1) and l1 - l0 > 0.5 and l1 > 1.5:
            out.append(j)
    return out


SUP = ["tr_sup_new", "tr_sup_cancel", "tr_sup_min"]


@trick
def support(ctx, rows):
    """T1 #3: the candidate team's other players commit to outlets when it wins the ball, and cancel the runs when it loses it."""
    m = _motion(ctx)
    for r in rows:
        s, t, k = r["s"], r["t"], r["k"]
        for key in SUP:
            r[key] = np.nan
        q = m.anchor(s)
        if q is None:
            continue
        now = _outlets(m, q, s, s + 15, k)
        before = _outlets(m, q, s - 25, s - 10, k)
        r["tr_sup_new"] = float(len(now) - len(before))
        cancel = 0
        e = min(m.n - 1, t + 20)
        for j in now:
            v0, v1 = m.v[min(t, m.n - 1), j], m.v[e, j]
            if np.isfinite(v0).all() and np.isfinite(v1).all():
                n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
                turned = n0 > 0.5 and n1 > 0.5 and v0 @ v1 / (n0 * n1) < 0.5
                if n0 - n1 > 1.5 or turned:
                    cancel += 1
        r["tr_sup_cancel"] = float(cancel)
        r["tr_sup_min"] = float(min(len(now), cancel))
    return SUP


def _shape(m, q, f0, f1, t):
    """Per frame, team t's far players (>= 20 m from q, at least 4): 20th percentile along its attack axis and width."""
    f0, f1 = m.clip(f0, f1)
    if q is None or f1 - f0 < 3:
        return np.nan, np.nan
    p = m.pm[f0:f1][:, m.out[t]]
    d = np.linalg.norm(p - q, axis=2)
    line, width = [], []
    for u in range(len(p)):
        sel = p[u][np.isfinite(d[u]) & (d[u] >= 20)]
        if len(sel) >= 4:
            line.append(np.percentile(m.sign[t] * sel[:, 0], 20))
            width.append(np.percentile(sel[:, 1], 90) - np.percentile(sel[:, 1], 10))
    return (float(np.mean(line)), float(np.mean(width))) if line else (np.nan, np.nan)


REM = ["tr_rem_%s%d" % (x, j) for x in ("d1", "d2", "dev") for j in range(4)]


@trick
def remote(ctx, rows):
    """T1 #5: far from the ball, the old attacking line brakes and the width pulses when the ball really changes hands."""
    m = _motion(ctx)
    for r in rows:
        s, t, k = r["s"], r["t"], r["k"]
        for key in REM:
            r[key] = np.nan
        if t - s < 20:
            continue
        q = m.anchor((s + t) // 2)
        if q is None:
            continue
        j = 0
        for team in (1 - k, k):
            pre, mid, post = _shape(m, q, s - 25, s - 5, team), _shape(m, q, s + 5, t - 5, team), _shape(m, q, t + 5, t + 35, team)
            for c in (0, 1):
                r["tr_rem_d1%d" % j], r["tr_rem_d2%d" % j] = mid[c] - pre[c], post[c] - mid[c]
                r["tr_rem_dev%d" % j] = mid[c] - 0.5 * (pre[c] + post[c])
                j += 1
    return REM


REL = ["tr_rel_ratio", "tr_rel_turn", "tr_rel_imp", "tr_rel_ak", "tr_rel_ao", "tr_rel_adv", "tr_rel_brake", "tr_rel_n"]


@trick
def release(ctx, rows):
    """T1 #6: prepared release versus passive rebound at the candidate team's last touch: ball speed in / out, turn, impulse, the
    outgoing ball's aim at a teammate versus an opponent, and the touching player's braking before it."""
    m = _motion(ctx)
    tt, tk = np.asarray(m.c["touch_t"]), np.asarray(m.c["touch_k"])
    for r in rows:
        s, t, k = r["s"], r["t"], r["k"]
        for key in REL:
            r[key] = np.nan
        idx = np.flatnonzero((tk == k) & (tt >= s - 6) & (tt < t + 6))
        r["tr_rel_n"] = float(len(idx))
        if not len(idx):
            continue
        c = int(tt[idx[-1]])
        inc, out = _ray(m.bxy, c - 16, c - 3), _ray(m.bxy, c + 4, c + 17)
        if inc is None or out is None:
            continue
        r["tr_rel_ratio"] = out[2] / max(inc[2], 0.5)
        cosang = float(np.clip(inc[1] @ out[1], -1, 1))
        r["tr_rel_turn"] = float(np.degrees(np.arccos(cosang)))
        r["tr_rel_imp"] = float(np.linalg.norm(out[1] * out[2] - inc[1] * inc[2]))
        origin = m.bxy[min(c + 4, m.n - 1)]
        if not np.isfinite(origin[0]):
            origin = out[0] - out[1] * 0.1
        u = min(c + 4, m.n - 1)
        rel = m.pm[u] - origin
        dist = np.linalg.norm(rel, axis=1)
        aim = {}
        for team in (k, 1 - k):
            sel = (m.team == team) & np.isfinite(dist) & (dist >= 4) & (dist <= 25)
            if sel.any():
                ang = np.arccos(np.clip(rel[sel] @ out[1] / dist[sel], -1, 1))
                aim[team] = float(np.max(np.exp(-(ang / 0.2) ** 2)))
            else:
                aim[team] = 0.0
        r["tr_rel_ak"], r["tr_rel_ao"], r["tr_rel_adv"] = aim[k], aim[1 - k], aim[k] - aim[1 - k]
        d = np.linalg.norm(m.pm[c] - m.bxy[c], axis=1) if np.isfinite(m.bxy[c, 0]) else None
        if d is not None:
            d = np.where(m.team == k, d, np.inf)
            d[~np.isfinite(d)] = np.inf
            j = int(np.argmin(d))
            v0, v1 = m.v[max(c - 15, 0), j], m.v[c, j]
            if np.isfinite(v0).all() and np.isfinite(v1).all():
                r["tr_rel_brake"] = float(np.linalg.norm(v0) - np.linalg.norm(v1))
    return REL


ORP = ["tr_orp_free", "tr_orp_valid", "tr_orp_delay", "tr_orp_first", "tr_orp_kfirst"]


@trick
def orphan(ctx, rows):
    """T1 #7: winning the first contact is weak when nobody controls the second ball: free flight after it, how long until the
    candidate team is alone on the ball, and which team first is."""
    m = _motion(ctx)
    tt, tk = np.asarray(m.c["touch_t"]), np.asarray(m.c["touch_k"])
    near = np.minimum(m.dk[0], m.dk[1])
    for r in rows:
        s, t, k = r["s"], r["t"], r["k"]
        idx = np.flatnonzero((tk == k) & (tt >= s - 6) & (tt < t + 6))
        c = int(tt[idx[0]]) if len(idx) else s
        f0, f1 = m.clip(c, c + 50)
        seen = np.isfinite(m.bxy[f0:f1, 0])
        r["tr_orp_valid"] = float(seen.mean()) if f1 > f0 else np.nan
        r["tr_orp_free"] = float(np.mean(near[f0:f1][seen] > 3.5)) if seen.any() else np.nan
        uk, uo = _isolated(m, c, c + 50, k, hold=6), _isolated(m, c + 12, c + 62, 1 - k, hold=6)
        r["tr_orp_delay"] = (uk - c) / ea.FPS if uk is not None else 3.0
        uk2 = _isolated(m, c + 12, c + 62, k, hold=6)
        r["tr_orp_first"] = 1.0 if uk2 is not None and (uo is None or uk2 <= uo) else (-1.0 if uo is not None else 0.0)
        r["tr_orp_kfirst"] = float(uk2 is not None)
    return ORP


GATE = ["tr_gate_ab", "tr_gate_min", "tr_gate_abg", "tr_gate_abm"]


@trick
def gates(ctx, rows):
    """T4 #8: the conjunctions a shallow tree on 200 positives may miss: every model family supports team k AND the specialist sees
    delayed control AND a gain at the start."""
    mem = Ctx.members(ctx.src)
    seed = getattr(ctx, "seed", 0)
    lg = {fam: [ea.logit(np.asarray(run[ctx.i], np.float64)) for run in runs] for fam, runs in mem.items()}
    o = np.asarray(exp_kpu.outputs("C1", ctx.src)[ss.SEEDS[seed]][ctx.i], np.float32)
    for r in rows:
        s, t, k = r["s"], r["t"], r["k"]
        sign = 1.0 if k == 0 else -1.0
        meds = []
        for fam, runs in lg.items():
            vals = []
            for x in runs:
                flank = np.r_[x[max(0, s - 10):s], x[t:t + 10]]
                vals.append(sign * (float(np.mean(x[s:t])) - (float(np.mean(flank)) if len(flank) else 0.0)))
            meds.append(np.median(vals))
        a = 1.0 / (1.0 + np.exp(-min(meds)))
        pk = ss.pk(o, k)
        late = pk[s + 12:t]
        b = float(np.mean(late)) if len(late) else np.nan
        g = float(np.max(o[max(0, s - 3):s + 4, 3 + k]))
        r["tr_gate_ab"], r["tr_gate_min"] = a * b, min(a, b) if np.isfinite(b) else np.nan
        r["tr_gate_abg"], r["tr_gate_abm"] = a * b * g, a * b * (1.0 - r.get("miss", 0.0))
    return GATE


STOP = ["tr_stop_to", "tr_stop_edge", "tr_stop_next", "tr_stop_prev", "tr_stop_line", "tr_stop_gap"]


@trick
def stops(ctx, rows):
    """T2, cheap part: the verifier knows the time since the last stoppage but not the time to the next. Look ahead on the row's own
    decode: seconds to the next stoppage, does the candidate touch it, does team k restart after it, did team k restart before it,
    how near the line the ball last was before it, how long play stays dead."""
    c = ctx.cues
    L, W = c["L"], c["W"]
    for r in rows:
        pr = ctx.bases[r["_dead"]]
        s, t, k = r["s"], r["t"], r["k"]
        for key in STOP:
            r[key] = np.nan
        dead = np.flatnonzero(pr[t:] == 2)
        if not len(dead):
            continue
        d0 = t + int(dead[0])
        r["tr_stop_to"] = min((d0 - t) / ea.FPS, 60.0)
        r["tr_stop_edge"] = float(r["e"] == d0)
        live = np.flatnonzero(pr[d0:] < 2)
        if len(live):
            d1 = d0 + int(live[0])
            r["tr_stop_next"] = float(pr[d1] == k)
            r["tr_stop_gap"] = (d1 - d0) / ea.FPS
        before = np.flatnonzero(pr[:s] == 2)
        if len(before):
            b0 = int(before[-1]) + 1
            if b0 < len(pr) and pr[b0] < 2:
                r["tr_stop_prev"] = float(pr[b0] == k)
        seen = np.flatnonzero(np.isfinite(c["bxy"][max(0, d0 - 50):d0 + 1, 0]))
        if len(seen):
            x, y = c["bxy"][max(0, d0 - 50) + int(seen[-1])]
            r["tr_stop_line"] = float(min(x, L - x, y, W - y))
    return STOP


# -- group E: does the specialist's evidence survive small changes of its input? (T4 #1) -----------------------------------

COPIES = ("j25a", "j25b", "j50a", "j50b", "smooth3", "mirror")


def _ar1(rng, shape, sigma):
    z = rng.standard_normal(shape)
    out = np.zeros(shape)
    out[0] = sigma * z[0]
    for u in range(1, shape[0]):
        out[u] = 0.82 * out[u - 1] + np.sqrt(1 - 0.82 ** 2) * sigma * z[u]
    return out


def perturbed(b, name, seed=20261104):
    """A copy of a half with the ball (and slightly the players) moved. Coordinates are normalised to the pitch."""
    pitch = np.array(b["pitch"], np.float64)
    P, B = np.asarray(b["P"], np.float64).copy(), np.asarray(b["B"], np.float64).copy()
    rng = np.random.default_rng(seed + COPIES.index(name))
    if name == "mirror":
        P[:, :, 1], B[:, 1] = 1.0 - P[:, :, 1], 1.0 - B[:, 1]
    elif name == "smooth3":
        ok = np.isfinite(B[:, 0])
        inner = ok[:-2] & ok[1:-1] & ok[2:]
        B[1:-1][inner] = (B[:-2][inner] + B[1:-1][inner] + B[2:][inner]) / 3
    else:
        sigma = float(name[1:3]) / 100.0
        B += _ar1(rng, B.shape, sigma) / pitch
        P += _ar1(rng, (P.shape[0], P.shape[1] * 2), 0.10).reshape(P.shape) / pitch
    return P.astype(np.float32), B.astype(np.float32)


def survival_outputs(src, seed=0):
    """Live-normalised P(home) of the specialist of v10 seed k on every perturbed copy of the half (CPU), cached per source."""
    path = os.path.join(ss.OUT, "survival_%s_s%d.pkl" % (src.replace("@", "-"), seed))
    if os.path.isfile(path):
        return bench._load(path)
    model = ss.load(os.path.join(ss.OUT, "models", "kpu_C1_s%d.pt" % ss.SEEDS[seed]))
    out = {}
    for name in COPIES:
        res = []
        for b in bench.blocks(src):
            P, B = perturbed(b, name)
            o = ss.predict(model, ss.features(P, B, b["team"], b["gk"], b["pitch"]))
            res.append(ss.pk(o, 0).astype(np.float16))
        out[name] = res
        print("    %s %s done" % (src, name), flush=True)
    bench._dump(out, path)
    return out


SUR = ["tr_sur_p10", "tr_sur_std", "tr_sur_pos", "tr_sur_mirror", "tr_sur_mean"]


@trick
def survival(ctx, rows):
    """T4 #1: the specialist's margin for team k inside the candidate (minus its flanks) over the perturbed copies: 10th percentile,
    spread, share still in favour, and the mirrored copy's disagreement with the plain output."""
    seed = getattr(ctx, "seed", 0)
    copies = survival_outputs(ctx.src, seed)
    plain = ea.logit(ss.pk(np.asarray(exp_kpu.outputs("C1", ctx.src)[ss.SEEDS[seed]][ctx.i], np.float32), 0))
    lg = {n: ea.logit(np.asarray(copies[n][ctx.i], np.float64)) for n in COPIES}

    def margin(x, r):
        s, t = r["s"], r["t"]
        flank = np.r_[x[max(0, s - 10):s], x[t:t + 10]]
        e = float(np.mean(x[s:t])) - (float(np.mean(flank)) if len(flank) else 0.0)
        return e if r["k"] == 0 else -e

    for r in rows:
        e = np.array([margin(lg[n], r) for n in COPIES])
        r["tr_sur_p10"], r["tr_sur_std"] = float(np.percentile(e, 10)), float(np.std(e))
        r["tr_sur_pos"], r["tr_sur_mean"] = float(np.mean(e > 0)), float(np.mean(e))
        r["tr_sur_mirror"] = abs(float(e[COPIES.index("mirror")]) - margin(plain, r))
    return SUR


def _strict(lab, r):
    sp = [(a, b) for a, b, k in ms.spells(lab) if k == r["k"] and b > r["s"] and a < r["t"]]
    return any((min(b, r["t"]) - max(a, r["s"])) / (max(b, r["t"]) - min(a, r["s"])) >= 0.5 for a, b in sp)


def _f1(blocks, labs, base, rows, p):
    by = {}
    for n, r in enumerate(rows):
        by.setdefault(r["i"], []).append(n)
    preds = [ev.apply(base[i], [rows[n] for n in by.get(i, [])], p[by.get(i, [])], True, True, 0.65, 0.35)
             for i in range(len(blocks))]
    return float(np.mean(list(er.by_match(blocks, labs, preds).values())))


def screen(names, seed=0, vseed=0):
    params = dict(ev.PARAMS, seed=vseed, deterministic=True, force_row_wise=True,
                  num_threads=int(os.environ.get("VERIFY_THREADS", "8")))
    base_name = os.environ.get("TRICK_BASE", BASE)
    exp_kpu.use(base_name)
    data = {(src, d): ev._rows4(src, seed, d) for src in exp_kpu.SRCS for d in ev.DEADS}
    for (src, d), v in data.items():
        for r in v["rows"]:
            r["_dead"] = d
    keys = exp_kpu.feats(base_name)
    new = []
    for src in exp_kpu.SRCS:
        blocks = bench.blocks(src)
        for i, b in enumerate(blocks):
            rows = [r for d in ev.DEADS for r in data[(src, d)]["rows"] if r["i"] == i]
            ctx = Ctx(src, i, b)
            ctx.seed = seed
            ctx.bases = {d: np.asarray(data[(src, d)]["base"][i]) for d in ev.DEADS}
            for name in names:
                got = TRICKS[name](ctx, rows)
                new += [k for k in got if k not in new]
        bench._blocks.clear()
        print("  features computed: %s" % src, flush=True)
    rows = [r for d in ev.DEADS for src in exp_kpu.SRCS for r in data[(src, d)]["rows"]]
    for r in rows:
        for k in new:
            r.setdefault(k, np.nan)
    p0 = ev.predict_lomo(rows, keys, params)
    p1 = ev.predict_lomo(rows, keys + new, params)
    n0 = 0
    print("\n%s, %d new features, v10 seed %d, verifier seed %d" % ("+".join(names), len(new), seed, vseed))
    for d in ev.DEADS:
        for src in exp_kpu.SRCS:
            part = data[(src, d)]["rows"]
            idx = np.arange(n0, n0 + len(part))
            n0 += len(part)
            blocks = bench.blocks(src)
            convs = SCORE[src]
            labs = {c: [np.asarray(g) for g in bench.labels(src, c)] for c in convs}
            y = np.array([r["y"] for r in part])
            ins = np.array([r["typ"] == "ins" for r in part])
            q0, q1 = p0[idx], p1[idx]
            strict = np.array([_strict(labs[convs[0]][r["i"]], r) if r["typ"] == "ins" else False for r in part])
            txt = "%-10s %-9s ins AUC %.3f -> %.3f, strict %.3f -> %.3f" % (
                src.split("@")[0], d, ea.auc(q0[ins & (y == 1)], q0[ins & (y == 0)]), ea.auc(q1[ins & (y == 1)], q1[ins & (y == 0)]),
                ea.auc(q0[ins & strict], q0[ins & ~strict]), ea.auc(q1[ins & strict], q1[ins & ~strict]))
            for c in convs:
                a, b = _f1(blocks, labs[c], data[(src, d)]["base"], part, q0), _f1(blocks, labs[c], data[(src, d)]["base"], part, q1)
                txt += " | %s %.4f %+.4f" % (c, a, b - a)
            print(txt, flush=True)
            bench._blocks.clear()
            bench._labels.clear()
    exp_kpu.use("v14")


# -- does a trick carry signal beyond the verifier? A small correction instead of more trees (T4 #2 / #3 idea) ------------

def corrected(rows, p0, new, seed=0, c=0.1):
    """Leave-one-match-out logistic correction of the insert rows: logit(p0) plus the new features (standardised, median-imputed,
    with missing flags), strong L2; deletion rows keep p0. New features cannot dilute the verifier's trees this way."""
    from sklearn.linear_model import LogisticRegression
    out = p0.copy()
    idx = np.array([n for n, r in enumerate(rows) if r["typ"] == "ins"])
    y = np.array([rows[n]["y"] for n in idx])
    g = np.array([rows[n]["group"] for n in idx])
    x = np.array([[rows[n].get(k, np.nan) for k in new] for n in idx], float)
    x[~np.isfinite(x)] = np.nan
    lg = ea.logit(p0[idx])
    for grp in np.unique(g):
        tr, te = (g != grp) & (y >= 0), g == grp
        med = np.nanmedian(x[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        miss = np.isnan(x)
        z = np.where(miss, med, x)
        mu, sd = z[tr].mean(0), z[tr].std(0) + 1e-6
        z = (z - mu) / sd
        keep = [j for j in range(miss.shape[1]) if 0.02 < miss[tr, j].mean() < 0.98]
        Z = np.column_stack([lg, z] + ([miss[:, keep].astype(float)] if keep else []))
        m = LogisticRegression(C=c, max_iter=500)
        m.fit(Z[tr], y[tr])
        out[idx[te]] = m.predict_proba(Z[te])[:, 1]
    return out


def signal(names, seed=0, vseed=0, score=None):
    """Every trick alone: strict insert AUC and spell F1 with the small correction (accepting as many inserts as v15, over all
    sources), plus the features' own AUC on DFL + Metrica, overall and among the candidates the verifier hesitates on (p0 >= 0.3)."""
    score = score or PRIMARY
    params = dict(ev.PARAMS, seed=vseed, deterministic=True, force_row_wise=True, num_threads=int(os.environ.get("VERIFY_THREADS", "8")))
    exp_kpu.use(BASE)
    data = {(src, d): ev._rows4(src, seed, d) for src in exp_kpu.SRCS for d in ev.DEADS}
    add_iou(data)
    for (src, d), v in data.items():
        for r in v["rows"]:
            r["_dead"] = d
    feats = {}
    for src in exp_kpu.SRCS:
        blocks = bench.blocks(src)
        for i, b in enumerate(blocks):
            rows = [r for d in ev.DEADS for r in data[(src, d)]["rows"] if r["i"] == i]
            ctx = Ctx(src, i, b)
            ctx.seed = seed
            ctx.bases = {d: np.asarray(data[(src, d)]["base"][i]) for d in ev.DEADS}
            for name in names:
                feats[name] = TRICKS[name](ctx, rows)
        bench._blocks.clear()
        print("  features computed: %s" % src, flush=True)
    keys = exp_kpu.feats(BASE)
    rows = [r for d in ev.DEADS for src in exp_kpu.SRCS for r in data[(src, d)]["rows"]]
    p0 = ev.predict_lomo(rows, keys, params)
    for r in rows:
        for fs in feats.values():
            for k in fs:
                r.setdefault(k, np.nan)
    runs = dict((n, [n]) for n in names)
    if len(names) > 1:
        runs["ALL"] = list(names)
    for label, group in runs.items():
        new = [k for n in group for k in feats[n]]
        q = corrected(rows, p0, new, vseed)
        ti, td = _count_thresholds(p0, q, rows, 1.0)
        n0 = 0
        txt = []
        for d in ev.DEADS:
            for src in exp_kpu.SRCS:
                part = data[(src, d)]["rows"]
                idx = np.arange(n0, n0 + len(part))
                n0 += len(part)
                if src not in score:
                    continue
                blocks = bench.blocks(src)
                ins = np.array([r["typ"] == "ins" for r in part])
                st = np.array([r["iou"] >= 0.5 for r in part])
                by = {}
                for n, r in enumerate(part):
                    by.setdefault(r["i"], []).append(n)
                base = data[(src, d)]["base"]
                res = []
                for pp, a, b in ((p0[idx], 0.65, 0.35), (q[idx], ti, 0.35)):
                    preds = [apply_ex(base[i], [part[n] for n in by.get(i, [])], pp[by.get(i, [])], t_ins=a, t_del=b) for i in range(len(blocks))]
                    res.append(float(np.mean(list(er.by_match(blocks, [np.asarray(g) for g in bench.labels(src, score[src][0])], preds).values()))))
                txt.append("%s %s AUC %.3f %+.3f F1 %+.4f" % (src[:3], d[0], ea.auc(p0[idx][ins & st], p0[idx][ins & ~st]),
                                                              ea.auc(q[idx][ins & st], q[idx][ins & ~st]) - ea.auc(p0[idx][ins & st], p0[idx][ins & ~st]), res[1] - res[0]))
        print("%-10s %s" % (label, " | ".join(txt)), flush=True)
    sel = np.array([r["typ"] == "ins" and not r["src"].startswith("sc") for r in rows])
    yy = np.array([r["iou"] >= 0.5 for r in rows])
    zone = p0 >= 0.3
    print("\nfeature AUC on DFL + Metrica inserts: all / p0 >= 0.3 (%d rows, %d true)" % ((sel & zone).sum(), (sel & zone & yy).sum()))
    for name in names:
        out = []
        for k in feats[name]:
            x = np.array([r[k] for r in rows], float)
            a = ea.auc(x[sel & yy], x[sel & ~yy])
            b = ea.auc(x[sel & zone & yy], x[sel & zone & ~yy])
            out.append((abs(b - 0.5) if np.isfinite(b) else 0, k, a, b))
        out.sort(reverse=True)
        print("  %-9s %s" % (name, "  ".join("%s %.3f/%.3f" % (k.replace("tr_", ""), a, b) for _, k, a, b in out[:5])), flush=True)
    exp_kpu.use("v14")


# -- group A: decision rules on the stored verifier probabilities (nothing retrained) -----------------------------------

Q = (0.3, 0.4, 0.5, 0.6, 0.7)
PRIMARY = {"dfl@v3": ("evctrl",), "metrica@v3": ("ctrl",)}


def _votes(z):
    """Share of the IoU thresholds a spell with IoU z passes."""
    return sum(z >= q for q in Q) / len(Q)


def _ell(left, right, length):
    """Thresholds the parent loses when its largest fragment is what is left of it (T3 #1)."""
    return 1.0 - _votes(max(left, right) / max(length, 1))


def apply_ex(pred, rows, p, t_ins=0.65, t_del=0.35, by_p=False, move=0, move_p=None):
    """exp_verify.apply with options. by_p: inserts in order of p, so that overlapping candidates compete (T4 #6). move: an
    accepted insert whose far fragment is at most `move` frames and borders a spell of team k moves that boundary instead of
    adding two possessions (T3 #5); move_p: its own, lower threshold. t_ins / t_del may be arrays per row."""
    out = pred.copy()
    ti = np.broadcast_to(np.asarray(t_ins, float), (len(rows),))
    td = np.broadcast_to(np.asarray(t_del, float), (len(rows),))
    for n, r in enumerate(rows):
        if r["typ"] != "del" or p[n] >= td[n]:
            continue
        s, t, k = r["s"], r["t"], r["k"]
        prev = out[s - 1] if s > 0 else 3
        nxt = out[t] if t < len(out) else 3
        if (prev == 1 - k and nxt in (1 - k, 2)) or (nxt == 1 - k and prev == 2):
            out[s:t] = 1 - k
    ins = [n for n, r in enumerate(rows) if r["typ"] == "ins"]
    if by_p:
        ins.sort(key=lambda n: -p[n])
    floor = ti if move_p is None else np.minimum(ti, move_p)
    for n in ins:
        if p[n] < floor[n]:
            continue
        r = rows[n]
        s, t, k = r["s"], r["t"], r["k"]
        if not np.all(out[s:t] == 1 - k):
            continue
        left, right, a, e = s - r["a"], r["e"] - t, r["a"], r["e"]
        if move and 0 < right <= move and e < len(out) and out[e] == k and np.all(out[t:e] == 1 - k):
            out[s:e] = k
            continue
        if move and 0 < left <= move and a > 0 and out[a - 1] == k and np.all(out[a:s] == 1 - k):
            out[a:t] = k
            continue
        if p[n] < ti[n] or (0 < left < r["min_len"]) or (0 < right < r["min_len"]):
            continue
        out[s:t] = k
    return out


def geometry_ins(rows, kappa, f=0.8):
    """T3 #1: the insert threshold moved by the arithmetic price of the fragments it leaves, scaled by kappa (the verifier's
    p is not calibrated to the metric, so the formula sets the shape and not the level)."""
    out = np.full(len(rows), 0.65)
    for n, r in enumerate(rows):
        if r["typ"] == "ins":
            ell = _ell(r["s"] - r["a"], r["e"] - r["t"], r["e"] - r["a"])
            out[n] = 0.65 + kappa * ((f + ell) / (2 + 2 * ell) - f / 2)
    return out


def geometry_del(rows, kappa, f=0.8):
    """T3 #2: delete if p < (F d / 2 + b_F) / (a_T + b_F), interior d = 2, a_T = 2 + ell, b_F = ell; next to a stoppage the level
    stays. Same remark on kappa: 0.35 keeps the level, the formula sets the differences."""
    out = np.full(len(rows), 0.35)
    for n, r in enumerate(rows):
        if r["typ"] == "del" and not (r.get("prev_dead", 0) or r.get("next_dead", 0)):
            a, b, w = r.get("prev_len", 0), r.get("next_len", 0), (r["t"] - r["s"]) / ea.FPS
            ell = _ell(a, b, a + b + w)
            out[n] = 0.35 + kappa * ((f + ell) / (2 + 2 * ell) - f / 2)
    return out


def capped(pred, rows, p, per_half, floor=0.5):
    """T4 #7: at most per_half inserts per half, best p first, each at least floor; deletions as v15."""
    out = apply_ex(pred, rows, p, t_ins=2.0)
    ins = sorted((n for n, r in enumerate(rows) if r["typ"] == "ins" and p[n] >= floor), key=lambda n: -p[n])
    done = 0
    for n in ins:
        if done >= per_half:
            break
        r = rows[n]
        left, right = r["s"] - r["a"], r["e"] - r["t"]
        if np.all(out[r["s"]:r["t"]] == 1 - r["k"]) and not (0 < left < r["min_len"]) and not (0 < right < r["min_len"]):
            out[r["s"]:r["t"]] = r["k"]
            done += 1
    return out


def shifted(rows, p, ds, dt, t_ins=0.65):
    """T3 #3: accepted inserts with their start moved by ds and their end by dt frames, inside the parent."""
    out = []
    for n, r in enumerate(rows):
        if r["typ"] == "ins" and p[n] >= t_ins:
            r2 = dict(r, s=max(r["a"], r["s"] + ds), t=min(r["e"], r["t"] + dt))
            r = r2 if r2["t"] - r2["s"] >= 3 else r
        out.append(r)
    return out


def policies():
    pol = {"v15": lambda b, r, p: apply_ex(b, r, p)}
    for kappa in (1, 3, 5):
        pol["A1 frag thr x%d" % kappa] = lambda b, r, p, kp=kappa: apply_ex(b, r, p, t_ins=geometry_ins(r, kp))
        pol["A2 del thr x%d" % kappa] = lambda b, r, p, kp=kappa: apply_ex(b, r, p, t_del=geometry_del(r, kp))
    for x in (25, 50):
        pol["A4 move <=%d" % x] = lambda b, r, p, m=x: apply_ex(b, r, p, move=m)
        pol["A4 move <=%d p.5" % x] = lambda b, r, p, m=x: apply_ex(b, r, p, move=m, move_p=0.5)
    pol["A5 overlap by p"] = lambda b, r, p: apply_ex(b, r, p, by_p=True)
    for n in (1, 2, 3):
        pol["A6 cap %d per half" % n] = lambda b, r, p, m=n: capped(b, r, p, m)
    for ds, dt in ((-4, 4), (-8, 8), (4, -4), (-4, 0), (0, 4), (4, 0), (0, -4), (-8, 0), (0, 8), (4, 4), (-4, -4)):
        pol["A3 ends %+d %+d" % (ds, dt)] = lambda b, r, p, x=ds, y=dt: apply_ex(b, shifted(r, p, x, y), p)
    return pol


def stored(seed, vseed):
    """{(src, dead): (rows, base, p)} of v15's research caches (under exp_kpu.use(BASE))."""
    out = {}
    for d in ev.DEADS:
        pp = bench._load(ev._path("probs4", seed, d, "q%d" % vseed))["G1"]
        n0 = 0
        for src in exp_kpu.SRCS:
            v = ev._rows4(src, seed, d)
            out[(src, d)] = (v["rows"], v["base"], pp[n0:n0 + len(v["rows"])])
            n0 += len(v["rows"])
    return out


def _table(cells, names, contexts):
    print("\n%-20s %s" % ("", " ".join("%-17s" % ("%s %s" % (s.split("@")[0], c)) for s, c in contexts)))
    print("%-20s %s" % ("", " ".join("%-17s" % "learned / true" for _ in contexts)))
    for name in names:
        row = []
        for s, c in contexts:
            pair = []
            for d in ev.DEADS:
                a = np.mean(cells[(s, c, d, names[0])])
                pair.append(a if name == names[0] else np.mean(cells[(s, c, d, name)]) - a)
            row.append(("%.4f / %.4f" if name == names[0] else "%+.4f / %+.4f") % tuple(pair))
        print("%-20s %s" % (name, " ".join("%-17s" % x for x in row)), flush=True)


def arith(seeds=(0, 1, 2), vseed=0, score=PRIMARY):
    exp_kpu.use(BASE)
    pol = policies()
    contexts = [(s, c) for s, cs in score.items() for c in cs]
    cells = {}
    blocks = {s: bench.blocks(s) for s in score}
    labs = {(s, c): [np.asarray(g) for g in bench.labels(s, c)] for s, c in contexts}
    for seed in seeds:
        st = stored(seed, vseed)
        for (src, d), (rows, base, p) in st.items():
            if src not in score:
                continue
            by = {}
            for n, r in enumerate(rows):
                by.setdefault(r["i"], []).append(n)
            parts = [([rows[n] for n in by.get(i, [])], p[by.get(i, [])]) for i in range(len(base))]
            for name, fn in pol.items():
                preds = [fn(base[i], parts[i][0], parts[i][1]) for i in range(len(base))]
                for c in score[src]:
                    cells.setdefault((src, c, d, name), []).append(np.mean(list(er.by_match(blocks[src], labs[(src, c)], preds).values())))
        print("  seed %d done" % seed, flush=True)
    _table(cells, list(pol), contexts)
    exp_kpu.use("v14")


# -- group B: what the verifier learns (same rows, same features) ---------------------------------------------------------

TARGET_CONV = {"dfl@v3": "evctrl", "metrica@v3": "ctrl", "sc@v3": "evctrl"}


def add_iou(data):
    """r["iou"]: best IoU of the candidate with a true spell of its team, in the target convention."""
    import exp_short_cands as ec
    for src in exp_kpu.SRCS:
        labs = [np.asarray(g) for g in bench.labels(src, TARGET_CONV[src])]
        for d in ev.DEADS:
            for r in data[(src, d)]["rows"]:
                r["iou"] = ec.best_iou(labs[r["i"]], r["s"], r["t"], r["k"])


def lomo_target(rows, keys, kind, seed=0):
    """Leave-one-match-out scores per candidate type. kind: strict (IoU >= 0.5), votes (share of the five IoU thresholds passed,
    soft labels), rank (lambdarank inside a match on the same five grades), votes_w (votes, positives of a match sharing a true
    spell not reweighted, candidates near the metric's upper thresholds weighted up)."""
    import lightgbm as lgb
    base = dict(ev.PARAMS, seed=seed, deterministic=True, force_row_wise=True, num_threads=int(os.environ.get("VERIFY_THREADS", "8")))
    p = np.full(len(rows), np.nan)
    for typ in ("ins", "del"):
        idx = np.array([n for n, r in enumerate(rows) if r["typ"] == typ])
        X = np.array([[rows[n][k] for k in keys] for n in idx], float)
        iou = np.array([rows[n]["iou"] for n in idx])
        votes = np.array([_votes(z) for z in iou])
        g = np.array([rows[n]["group"] for n in idx])
        for grp in np.unique(g):
            tr, te = g != grp, g == grp
            if kind == "strict":
                par, ds = base, lgb.Dataset(X[tr], (iou[tr] >= 0.5).astype(float))
            elif kind == "votes":
                par, ds = dict(base, objective="cross_entropy"), lgb.Dataset(X[tr], votes[tr])
            elif kind == "rank":
                order = np.argsort(g[tr], kind="stable")
                _, sizes = np.unique(g[tr][order], return_counts=True)
                par = dict(base, objective="lambdarank", metric="ndcg", min_data_in_leaf=20)
                ds = lgb.Dataset(X[tr][order], np.round(5 * votes[tr][order]).astype(int), group=sizes)
            else:
                raise ValueError(kind)
            p[idx[te]] = lgb.train(par, ds, ev.ROUNDS).predict(X[te])
    return p


def _count_thresholds(p_ref, p_new, rows, factor=1.0):
    """Thresholds of the new score that accept factor x as many inserts and as many deletions as v15's 0.65 / 0.35: chosen on
    counts over all sources, never on a score."""
    ins = np.array([r["typ"] == "ins" for r in rows])
    n_ins = int(round(factor * (p_ref[ins] >= 0.65).sum()))
    n_del = int((p_ref[~ins] < 0.35).sum())
    a = np.sort(p_new[ins])[::-1]
    b = np.sort(p_new[~ins])
    return a[min(n_ins, len(a) - 1)] if n_ins else np.inf, b[min(n_del, len(b) - 1)] if n_del else -np.inf


def target(kinds, seed=0, vseed=0, factors=(1.0, 0.75, 1.5), score=PRIMARY):
    params = dict(ev.PARAMS, seed=vseed, deterministic=True, force_row_wise=True, num_threads=int(os.environ.get("VERIFY_THREADS", "8")))
    exp_kpu.use(BASE)
    data = {(src, d): ev._rows4(src, seed, d) for src in exp_kpu.SRCS for d in ev.DEADS}
    add_iou(data)
    keys = exp_kpu.feats(BASE)
    for d in ev.DEADS:
        rows = [r for src in exp_kpu.SRCS for r in data[(src, d)]["rows"]]
        p0 = ev.predict_lomo(rows, keys, params)
        ref = {"v15": p0}
        for k in kinds:
            ref[k] = lomo_target(rows, keys, k, vseed)
        n0 = 0
        for src in exp_kpu.SRCS:
            part = data[(src, d)]["rows"]
            idx = np.arange(n0, n0 + len(part))
            n0 += len(part)
            if src not in score:
                continue
            blocks = bench.blocks(src)
            ins = np.array([r["typ"] == "ins" for r in part])
            strict = np.array([r["iou"] >= 0.5 for r in part])
            by = {}
            for n, r in enumerate(part):
                by.setdefault(r["i"], []).append(n)
            for c in score[src]:
                labs = [np.asarray(g) for g in bench.labels(src, c)]
                for name, q in ref.items():
                    qq = q[idx]
                    auc = ea.auc(qq[ins & strict], qq[ins & ~strict])
                    txt = "%-8s %-9s %-7s strict ins AUC %.3f" % (src.split("@")[0], d, name, auc)
                    for fac in ((1.0,) if name == "v15" else factors):
                        ti, td = (0.65, 0.35) if name == "v15" else _count_thresholds(p0, q, rows, fac)
                        preds = [apply_ex(data[(src, d)]["base"][i], [part[n] for n in by.get(i, [])], qq[by.get(i, [])], t_ins=ti, t_del=td)
                                 for i in range(len(blocks))]
                        f1 = float(np.mean(list(er.by_match(blocks, labs, preds).values())))
                        txt += " | x%.2f %.4f" % (fac, f1)
                    print(txt, flush=True)
    exp_kpu.use("v14")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "screen":
        screen(sys.argv[2].split(","), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif sys.argv[1] == "signal":
        signal(sys.argv[2].split(","), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif sys.argv[1] == "arith":
        arith()
    elif sys.argv[1] == "target":
        target(sys.argv[2].split(","), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
