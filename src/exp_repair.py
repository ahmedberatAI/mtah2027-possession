"""Round 84 (docs/RAPOR.md): v21's whole team side on tracking given as arrays, so that a repaired or a perturbed input can be compared with the original. Final models, on the CPU, labels never read here. Research code; prints aggregates only.

    python -I src/exp_repair.py target base     # the original input through the same path: must give v21's labels
    python -I src/exp_repair.py target fix      # the repaired team labels (src/team_color.py)
    python -I src/exp_repair.py compare         # how much the output changes, aggregates only
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["BF_WIN"] = "300"
os.environ.pop("BF_TAG", None)
_argv = list(sys.argv)
sys.argv = [sys.argv[0], os.path.dirname(HERE), "none"]
import atd  # noqa: E402
import bench  # noqa: E402
import bfree  # noqa: E402
import exp_ballfree as bf  # noqa: E402
import exp_edges  # noqa: E402,F401  registers the sim variants
import exp_short as es  # noqa: E402
import model_v7  # noqa: E402
import model_v9  # noqa: E402
import model_v10  # noqa: E402
import model_v15  # noqa: E402
import model_v17  # noqa: E402
import model_v20  # noqa: E402
import model_v21  # noqa: E402
import seq_pff as seq  # noqa: E402
import short_specialist as ss  # noqa: E402
import stack_model as sm  # noqa: E402
import team_color  # noqa: E402
import torch  # noqa: E402
sys.argv = _argv
warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(HERE), "cache", "team_color")
GRU_DIR = os.path.join(os.path.dirname(HERE), "models", "bigru_dfl_pff_v1")


def features_of(blocks, tag, procs=6):
    from multiprocessing import Pool
    folder = os.path.join(OUT, "feat_" + tag)
    os.makedirs(folder, exist_ok=True)
    outs = [os.path.join(folder, b["key"].replace("/", "_") + ".pkl") for b in blocks]
    todo = [(model_v7.VERSION, b, o) for b, o in zip(blocks, outs) if not os.path.isfile(o)]
    if todo:
        with Pool(min(procs, len(todo))) as pool:
            pool.map(bench._block_feats, todo)
    return [bench._load(o) for o in outs]


def gru_members():
    import json
    manifest = json.load(open(os.path.join(GRU_DIR, "manifest.json"), encoding="utf-8"))
    out = []
    for item in manifest["models"]:
        ck = torch.load(os.path.join(GRU_DIR, item["file"]), map_location="cpu", weights_only=False)
        c = ck["config"]
        net = seq.Net(345, c["kind"], c["width"], c["drop"])
        net.load_state_dict(ck["state"])
        out.append((net.eval(), seq.Scaler(ck["mean"], ck["scale"])))
    return out


def channels(blocks, tag, procs=6, threads=4):
    """Per half (v10 raw, BiGRU ensemble, J1's outputs, BF's q, BF's alive), final models, on the CPU."""
    path = os.path.join(OUT, "chan_%s.pkl" % tag)
    if os.path.isfile(path):
        return bench._load(path)
    torch.set_num_threads(threads)
    fs = features_of(blocks, tag, procs)
    print("  features ready", flush=True)
    team_model = model_v10.team_model()
    members = gru_members()
    j1 = ss.load(model_v17.CONFIG["model"])
    nets = []
    for seed in model_v21.BF_SEEDS:
        n_ = bf.build_net()
        n_.load_state_dict(torch.load(os.path.join(model_v21.BF_DIR, "bf_s%d.pt" % seed), map_location="cpu", weights_only=False)["state"])
        nets.append(n_.eval())
    res = []
    for b, f in zip(blocks, fs):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        v = bench.predict_team(team_model, dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
        g = np.mean([seq.predict_raw(net, sc, seq.views(f), len(f["XF"])).astype(np.float64) for net, sc in members], axis=0)
        o = ss.predict(j1, ss.features(b["P"], b["B"], b["team"], b["gk"], b["pitch"])).astype(np.float16)
        tok, _, signs = bf.tokens(b["P"], b["B"], b["team"], b["gk"], b["pitch"])
        r_ = [bf.predict(net, tok, signs, len(b["P"]), "cpu") for net in nets]
        lq = np.mean([sm.logit(z[0]) for z in r_], axis=0)
        res.append((np.asarray(v, np.float64), g, o, (1 / (1 + np.exp(-lq))).astype(np.float32), np.mean([z[1] for z in r_], axis=0).astype(np.float32)))
        print("  %s done" % b["key"], flush=True)
    bench._dump(res, path)
    return res


def team_side(blocks, chans, alive_list, stack_path=None):
    """Per half: the mix q of v21 and the alive evidence. alive_list: v20's alive evidence per half."""
    net = sm.load(stack_path or model_v21.STACK)
    q, al = [], []
    for b, (v, g, o, q_b, a_b), a in zip(blocks, chans, alive_list):
        x = sm.inputs(b, v, g, o, a, q_b, a_b)
        qs, alive = sm.score_half(net, x, len(v))
        blend = es.blend([v], [g], model_v15.W_BIGRU)[0]
        ph = model_v9.smooth((1 - model_v20.W_J1) * blend + model_v20.W_J1 * ss.pk(o, 0))
        n = min(len(ph), len(qs), len(a))
        mix = np.clip(1 / (1 + np.exp(-(model_v21.W_Q * sm.logit(qs[:n]) + (1 - model_v21.W_Q) * sm.logit(ph[:n])))), 1e-4, 1 - 1e-4)
        q.append(mix)
        al.append((1 - model_v21.W_A) * np.asarray(a[:n], np.float64) + model_v21.W_A * alive[:n])
    return q, al


def target_blocks(kind):
    bench._barath_inputs()
    out = []
    for (h, a, e), b in zip(atd.HALVES, bench.blocks("barath")):
        if kind == "fix":
            P2, team2, gk2 = team_color.repaired(b["P"], b["team"], b["gk"], a, h)
        else:
            P2 = np.full((b["P"].shape[0], 2 * b["P"].shape[1], 2), np.nan, np.float32)
            P2[:, :b["P"].shape[1]] = b["P"][:, :, :2]
            team2, gk2 = np.r_[b["team"], 1 - np.asarray(b["team"])], np.r_[b["gk"], np.zeros(len(b["gk"]), bool)]
        out.append(dict(b, P=P2, team=team2, gk=gk2, key="%s_%s" % (b["key"], kind)))
    return out


def target_run(kind):
    import make_submission
    blocks = target_blocks(kind)
    chans = channels(blocks, "target_" + kind)
    f = make_submission.merged_features()
    a_all = model_v20.alive_evidence(f)
    al_in = [a_all[b["a"]:b["b"] + 1] for b in blocks]
    q, al = team_side(blocks, chans, al_in)
    p = dict(model_v21.DEFAULT)
    sp = model_v7.spec(p)
    labels = np.full(f["n"], 3, np.int8)
    for b, q_, a_ in zip(blocks, q, al):
        q_ = np.clip(q_, 1e-4, 1 - 1e-4)
        S = np.stack([a_ / 2 + np.log(q_), a_ / 2 + np.log(1 - q_), -a_ / 2], 1)
        if b["a"] == atd.HALVES[1][1]:
            S[:atd.KICKOFF_2ND - b["a"], 2] += 1e6
        labels[b["a"]:b["a"] + len(S)] = sp.decode(S)
    np.save(os.path.join(OUT, "labels_%s.npy" % kind), labels)
    print("%s: labels written (not printed)" % kind, flush=True)


def inject(P, team, gk, seed, share, med_s=3.8, sigma=1.03):
    """The target's measured defect on an open half: per field player, stretches that follow the other team. Returns P2 (T, 2S, 2), team2, gk2 as team_color.repaired does, and the realised share."""
    rng = np.random.default_rng(seed)
    T, S = P.shape[0], P.shape[1]
    team = np.asarray(team)
    gk = np.asarray(gk, bool)
    wrong = np.zeros((T, S), bool)
    mean_len = med_s * 25 * np.exp(sigma ** 2 / 2)
    gap_mean = mean_len * (1 - share) / max(share, 1e-9)
    present = np.isfinite(P[:, :, 0])
    for s in np.flatnonzero(~gk):
        t = int(rng.exponential(gap_mean)) if share > 0 else T
        while t < T:
            d = max(1, int(rng.lognormal(np.log(med_s * 25), sigma)))
            wrong[t:t + d, s] = True
            t += d + int(rng.exponential(gap_mean))
    wrong &= present
    P2 = np.full((T, 2 * S, 2), np.nan, np.float32)
    keep = np.asarray(P[:, :, :2], np.float32).copy()
    keep[wrong] = np.nan
    P2[:, :S] = keep
    moved = np.full((T, S, 2), np.nan, np.float32)
    moved[wrong] = np.asarray(P[:, :, :2], np.float32)[wrong]
    P2[:, S:] = moved
    field = present & ~gk[None, :]
    return P2, np.r_[team, 1 - team].astype(team.dtype), np.r_[gk, np.zeros(S, bool)], wrong[field].mean()


DOSE_SETS = {"dfl": ("dfl@v3", "dfl@v3", "evctrl"), "metrica": ("metrica@v3g", "metrica@v3", "ctrl"), "sc": ("sc@v3", "sc@v3", "evctrl"), "st": ("st@v3", "st@v3", "evctrl")}
_cache = {}


def _models(threads):
    if not _cache:
        torch.set_num_threads(threads)
        _cache["team"] = model_v10.team_model()
        _cache["gru"] = gru_members()
        _cache["j1"] = ss.load(model_v17.CONFIG["model"])
        nets = []
        for seed in model_v21.BF_SEEDS:
            n_ = bf.build_net()
            n_.load_state_dict(torch.load(os.path.join(model_v21.BF_DIR, "bf_s%d.pt" % seed), map_location="cpu", weights_only=False)["state"])
            nets.append(n_.eval())
        _cache["bf"] = nets
    return _cache


def _half_job(job):
    """One half under one condition: channels, then cached. job: (block, path, share, seed)."""
    b, path, share, seed = job
    if os.path.isfile(path):
        return path
    warnings.filterwarnings("ignore")
    m = _models(int(os.environ.get("DOSE_THREADS", "2")))
    P2, team2, gk2, real = inject(np.asarray(b["P"]), b["team"], b["gk"], seed, share)
    bb = dict(b, P=P2, team=team2, gk=gk2)
    F, Th, Ta = bench.mlfeat.compute(bb["P"], bb["B"], bb["team"], bb["gk"], bb["pitch"])
    for fn in bench._feat_fns(model_v7.VERSION):
        f2, th2, ta2 = fn(bb["P"], bb["B"], bb["team"], bb["gk"], bb["pitch"])
        F.update(f2)
        Th.update(th2)
        Ta.update(ta2)
    fk, tk = sorted(F), sorted(Th)
    f = {"XF": np.stack([F[k] for k in fk], 1).astype(np.float32), "XH": np.stack([Th[k] for k in tk], 1).astype(np.float32), "XA": np.stack([Ta[k] for k in tk], 1).astype(np.float32), "fk": fk, "tk": tk}
    h, a = bfree.bf_feats(bb["P"], bb["team"], bb["gk"], bb["pitch"])
    v = bench.predict_team(m["team"], dict(XF=f["XF"], XH=np.hstack([f["XH"], h]), XA=np.hstack([f["XA"], a])))
    g = np.mean([seq.predict_raw(net, sc, seq.views(f), len(f["XF"])).astype(np.float64) for net, sc in m["gru"]], axis=0)
    o = ss.predict(m["j1"], ss.features(bb["P"], bb["B"], bb["team"], bb["gk"], bb["pitch"])).astype(np.float16)
    tok, _, signs = bf.tokens(bb["P"], bb["B"], bb["team"], bb["gk"], bb["pitch"])
    r_ = [bf.predict(net, tok, signs, len(bb["P"]), "cpu") for net in m["bf"]]
    lq = np.mean([sm.logit(z[0]) for z in r_], axis=0)
    bench._dump(dict(v=np.asarray(v, np.float32), g=np.asarray(g, np.float32), o=o, qb=(1 / (1 + np.exp(-lq))).astype(np.float32), ab=np.mean([z[1] for z in r_], axis=0).astype(np.float32), real=float(real)), path)
    return path


def dose_blocks(k):
    src, _, _ = DOSE_SETS[k]
    bl = bench.blocks(src)
    if k == "metrica":
        bl = [b for b in bl if b.get("seed", 0) == 0]    # one sim draw per game
    return bl


def dose_prep(sets, conds, workers=8):
    """conds: {name: share}. Channels of every half under every condition; aggregate prints only."""
    from multiprocessing import get_context
    jobs = []
    folder = os.path.join(OUT, "dose")
    os.makedirs(folder, exist_ok=True)
    for k in sets:
        for i, b in enumerate(dose_blocks(k)):
            for name, share in conds.items():
                seed = 8400 + 1000 * list(DOSE_SETS).index(k) + i
                jobs.append((b, os.path.join(folder, "%s_%d_%s.pkl" % (k, i, name)), share, seed))
    todo = [j for j in jobs if not os.path.isfile(j[1])]
    print("halves x conditions: %d, to do %d" % (len(jobs), len(todo)), flush=True)
    with get_context("spawn").Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_half_job, todo)):
            if i % 5 == 0:
                print("  %d / %d" % (i + 1, len(todo)), flush=True)


def dose_eval(sets, conds):
    """spellF1 of v21's path under each condition, final models, the frozen alive evidence of round 61."""
    import exp_crf as ec
    import exp_blend_weight  # noqa: F401
    from exp_crf import er
    sys.stdout.reconfigure(encoding="utf-8")
    pen, c24 = 6.0, 24.0
    for k in sets:
        src, lab_src, conv = DOSE_SETS[k]
        bl = dose_blocks(k)
        all_labs = [np.asarray(g) for g in bench.labels(lab_src, conv)]
        all_ev = bench._load(os.path.join(ec.OUT, "evidence_%s.pkl" % k))["ev"]
        if k == "metrica":
            pick = [i for i, b in enumerate(bench.blocks(src)) if b.get("seed", 0) == 0]
            all_labs, all_ev = [all_labs[i] for i in pick], [all_ev[i] for i in pick]
        res = {}
        for name in conds:
            chans, reals = [], []
            for i in range(len(bl)):
                d = bench._load(os.path.join(OUT, "dose", "%s_%d_%s.pkl" % (k, i, name)))
                chans.append((d["v"], d["g"], d["o"], d["qb"], d["ab"]))
                reals.append(d["real"])
            ns = [min(len(c[0]), len(e), len(g)) for c, e, g in zip(chans, all_ev, all_labs)]
            ev_ = [np.asarray(e[:m], np.float64) for e, m in zip(all_ev, ns)]
            bb = [dict(b, P=np.asarray(b["P"])[:m], B=np.asarray(b["B"])[:m]) for b, m in zip(bl, ns)]
            q, al = team_side(bb, [tuple(np.asarray(x)[:m] for x in c) for c, m in zip(chans, ns)], ev_)
            ns2 = [len(x) for x in q]
            tr = [g[:m] for g, m in zip(all_labs, ns2)]
            preds = ec.decode_all(al, q, pen, c24)
            res[name] = er.by_match(bl, tr, preds)
            print("%s %s, %s: realised share %.4f, spellF1 %.4f" % (k, conv, name, np.mean(reals), np.mean(list(res[name].values()))), flush=True)
        base = res["clean"]
        for name in conds:
            if name != "clean":
                diff = np.array([res[name][m_] - base[m_] for m_ in base])
                print("  %s - clean: %+.4f (matches worse %d / %d)" % (name, diff.mean(), (diff < 0).sum(), len(diff)), flush=True)


def spells(lab):
    lab = np.asarray(lab)
    out = []
    i = 0
    n = len(lab)
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        if lab[i] < 2:
            out.append((i, j, int(lab[i])))
        i = j
    return out


def match_share(a, b, thr):
    """Share of the spells of a with a same-team spell of b at IoU >= thr."""
    hit = 0
    for x, y, k in a:
        for u, w, m in b:
            if m != k or w <= x or u >= y:
                continue
            if (min(y, w) - max(x, u)) / (max(y, w) - min(x, u)) >= thr:
                hit += 1
                break
    return hit / max(len(a), 1)


def compare():
    sys.stdout.reconfigure(encoding="utf-8")
    codes = {"H": 0, "A": 1, "D": 2, "X": 3}
    with open(os.path.join(os.path.dirname(HERE), "submissions", "candidate_v21.csv")) as fh:
        next(fh)
        v21 = np.array([codes[ln.strip().split(",")[1]] for ln in fh], np.int8)
    g = atd.graded_mask()
    for kind in ("base", "fix"):
        path = os.path.join(OUT, "labels_%s.npy" % kind)
        if not os.path.isfile(path):
            continue
        z = np.load(path)
        print("%s vs v21: graded frames that differ %.5f (team %.5f, live/dead %.5f)" % (kind, np.mean(z[g] != v21[g]), np.mean((z[g] != v21[g]) & (z[g] < 2) & (v21[g] < 2)), np.mean((z[g] != v21[g]) & ((z[g] == 2) | (v21[g] == 2)))))
        s21, sz = [], []
        for _, a, e in atd.HALVES:
            s21 += [(x + a, y + a, k) for x, y, k in spells(v21[a:e + 1])]
            sz += [(x + a, y + a, k) for x, y, k in spells(z[a:e + 1])]
        d21 = np.array([y - x for x, y, _ in s21]) / 25
        dz = np.array([y - x for x, y, _ in sz]) / 25
        print("  possessions v21 %d (< 4 s %d), %s %d (< 4 s %d); spells of v21 with a same-team spell of %s at IoU >= .3 / .5 / .7: %.3f %.3f %.3f; the other way: %.3f %.3f %.3f" % (len(s21), (d21 < 4).sum(), kind, len(sz), (dz < 4).sum(), kind, match_share(s21, sz, .3), match_share(s21, sz, .5), match_share(s21, sz, .7), match_share(sz, s21, .3), match_share(sz, s21, .5), match_share(sz, s21, .7)))
        print("  home share of possessions' frames v21 %.4f, %s %.4f" % (np.mean(v21[g & (v21 < 2)] == 0), kind, np.mean(z[g & (z < 2)] == 0)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if cmd == "target":
        target_run(sys.argv[2])
    elif cmd == "compare":
        compare()
    elif cmd in ("dose", "doseval"):
        conds = {"clean": 0.0, "n103": 0.0103, "n030": 0.0030}
        sets = sys.argv[2].split(",")
        if cmd == "dose":
            dose_prep(sets, conds, int(os.environ.get("DOSE_WORKERS", "8")))
        else:
            dose_eval(sets, conds)
