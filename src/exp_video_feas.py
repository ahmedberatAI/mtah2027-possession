"""Round 22 feasibility of a video touch model (user: "ölçümü yap"). AGGREGATE output only -- no frames, no frame
numbers, no per-candidate rows.

Can video tell who touched the ball at the moments v14's short-possession verifier decides? For every candidate the
verifier scores on BAR-ATH (v14: blend .5, its own decode, inserts and deletes), over the candidate +-0.5 s:
  visible   the broadcast shows live play that agrees with the tactical tracking (src/broadcast.py b_live), its
            homography is valid and the ball (tactical ATD, broadcast fill, +-1 s carry) projects inside the image
  ball px   apparent ball diameter = 0.22 m x the homography's lateral ground scale (px per metre, the larger of the
            two pitch axes at the ball's spot) -- broadcast and tactical
  contested both teams have a player within 3 m of the ball near the candidate's start
Pre-registered go / no-go (round 22): >= 60% of candidates visible AND median broadcast ball diameter >= ~10 px.
  python -I src/exp_video_feas.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import atd  # noqa: E402
import bench  # noqa: E402
import bfree  # noqa: E402
import broadcast  # noqa: E402
import exp_short as es  # noqa: E402
import exp_short_angles as ea  # noqa: E402
import exp_smooth  # noqa: E402
import exp_verify as ev  # noqa: E402
import model_v7  # noqa: E402
import model_v10  # noqa: E402
import model_v14  # noqa: E402
import short_verifier  # noqa: E402
from make_submission import merged_features  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
L, W = atd.PITCH
PAD = 12                                   # frames around a candidate (0.5 s)


def homographies(source):
    h = pd.read_csv(atd.paths(source)["homography"])
    H = h[["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]].values.reshape(-1, 3, 3)
    return H, h["valid"].values.astype(bool)


def project(H, x, y):
    """Normalised pitch (x, y) -> pixels (u, v) and the lateral ground scale in px per metre."""
    p = np.einsum("nij,nj->ni", H, np.stack([x, y, np.ones_like(x)], 1))
    u, v, w = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2], p[:, 2]
    dux = (H[:, 0, 0] - u * H[:, 2, 0]) / w / L
    dvx = (H[:, 1, 0] - v * H[:, 2, 0]) / w / L
    duy = (H[:, 0, 1] - u * H[:, 2, 1]) / w / W
    dvy = (H[:, 1, 1] - v * H[:, 2, 1]) / w / W
    return u, v, np.maximum(np.hypot(dux, dvx), np.hypot(duy, dvy))


def carry(xy, n=25):
    """Fill NaN ball positions from the nearest finite one within n frames (forward, then backward)."""
    out = pd.DataFrame(xy).ffill(limit=n).bfill(limit=n).values
    return out


def video_size(source):
    import cv2
    path = next(v for k, v in atd.paths(source).items() if str(v).lower().endswith(".mp4"))
    cap = cv2.VideoCapture(str(path))
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return size


def candidates(f):
    """v14's verifier candidates on BAR-ATH: (start, end) in tactical frames, type, verifier probability."""
    labels = model_v7.predict(f, dict(model_v14.DEFAULT), probs_path=model_v14.ensure_probs())
    vk = model_v14.VERIFIER
    models = short_verifier.train(vk["use_g"], base="blend", w=vk["w"])
    team_model = model_v10.team_model()
    bench._barath_inputs()
    gru_all = bench._load(os.path.join(bench.ROOT, "cache", "shared_bigru", "barath_equal_probabilities.pkl"))
    out = []
    for b, ft in zip(bench.blocks("barath"), bench.features(model_v7.VERSION, "barath")):
        h, a = bfree.bf_feats(b["P"], b["team"], b["gk"], b["pitch"])
        raw = bench.predict_team(team_model, dict(XF=ft["XF"], XH=np.hstack([ft["XH"], h]), XA=np.hstack([ft["XA"], a])))
        raw = es.blend([raw], [gru_all[b["a"]][1]], vk["w"])[0]
        smo = exp_smooth.smooth([raw], 25)[0]
        pr = labels[b["a"]:b["b"] + 1].astype(np.int8)
        rows = ev.rows_for("barath", [b], [ea.frame_cues(b)], [pr], [pr], [smo], [raw], [gru_all[b["a"]][1]],
                           int(model_v7.DEFAULT["min_poss"]))
        for typ in ("ins", "del"):
            idx = [n for n, r in enumerate(rows) if r["typ"] == typ]
            if not idx:
                continue
            X = np.array([[rows[n][k] for k in short_verifier.keys(vk["use_g"])] for n in idx], float)
            for n, q in zip(idx, models[typ].predict(X)):
                out.append((b["a"] + rows[n]["s"], b["a"] + rows[n]["t"], typ, float(q)))
    return labels, out


def main():
    f = merged_features()
    labels, cands = candidates(f)
    n = len(labels)
    tac = atd.load("tactical")
    ball = np.asarray(tac["ball"][:, :2], np.float64)
    bb = np.asarray(f["b_ball"], np.float64) / np.array([L, W])           # broadcast fill (metres -> normalised)
    ball = np.where(np.isfinite(ball[:, :1]), ball, np.where(np.isfinite(bb[:, :1]), bb, np.nan))
    ball = carry(ball)
    ok = np.isfinite(ball[:, 0])
    # tactical image
    Ht, vt = homographies("tactical")
    tw, th = video_size("tactical")
    ut, wt, st = project(Ht[:n], np.nan_to_num(ball[:, 0]), np.nan_to_num(ball[:, 1]))
    tac_px = np.where(ok & vt[:n], 0.22 * st, np.nan)
    # broadcast image
    Hb, vb = homographies("broadcast")
    bw, bh = video_size("broadcast")
    t2b = broadcast.tactical_to_broadcast()
    have = t2b >= 0
    bi = np.where(have, t2b, 0)
    Hb_t = np.where(vb[bi][:, None, None], Hb[bi], np.eye(3))
    ub, wb, sb = project(Hb_t, np.nan_to_num(ball[:, 0]), np.nan_to_num(ball[:, 1]))
    inside = (ub >= 0) & (ub < bw) & (wb >= 0) & (wb < bh)
    live_b = np.asarray(f["b_live"], bool) & have & vb[bi]
    vis = live_b & ok & inside
    bro_px = np.where(vis, 0.22 * sb, np.nan)
    # contested: both teams within 3 m of the ball
    P = atd.to_metres(tac["players"][:, :, :2])
    bm = ball * np.array([L, W])
    d = np.linalg.norm(P - bm[:, None, :], axis=-1)
    near = [np.nanmin(np.where(tac["team"][None, :] == k, d, np.inf), axis=1) <= 3.0 for k in (0, 1)]
    contested = near[0] & near[1] & ok

    graded = atd.graded_mask(n)
    live = graded & (labels < 2)
    print("video sizes: tactical %dx%d, broadcast %dx%d" % (tw, th, bw, bh))
    print("all live play (v14 base decode, graded): broadcast visible %.3f | ball px broadcast median %.1f "
          "[p25 %.1f, p75 %.1f] | tactical median %.1f [p25 %.1f, p75 %.1f] | contested %.3f" % (
              vis[live].mean(), np.nanmedian(bro_px[live]), np.nanpercentile(bro_px[live], 25),
              np.nanpercentile(bro_px[live], 75), np.nanmedian(tac_px[live]), np.nanpercentile(tac_px[live], 25),
              np.nanpercentile(tac_px[live], 75), contested[live].mean()))

    def summary(name, sel):
        c = [cands[i] for i in np.flatnonzero(sel)]
        if not c:
            print("%-26s none" % name)
            return
        share, start_vis, bpx, tpx, cont = [], [], [], [], []
        for s, t, typ, q in c:
            w = slice(max(s - PAD, 0), min(t + PAD, n))
            share.append(vis[w].mean())
            start_vis.append(vis[max(s - 6, 0):s + 7].any())
            bpx.append(np.nanmedian(bro_px[w]) if np.isfinite(bro_px[w]).any() else np.nan)
            tpx.append(np.nanmedian(tac_px[w]) if np.isfinite(tac_px[w]).any() else np.nan)
            cont.append(contested[max(s - 12, 0):s + 13].any())
        share, bpx, tpx = np.array(share), np.array(bpx), np.array(tpx)
        visible = share >= 0.5
        print("%-26s n %4d | visible (>= half the window) %.3f | start visible %.3f | contested at start %.3f | "
              "ball px broadcast median %.1f [p25 %.1f] (>= 10 px: %.3f of visible) | tactical median %.1f [p25 %.1f]" % (
                  name, len(c), visible.mean(), np.mean(start_vis), np.mean(cont),
                  np.nanmedian(bpx[visible]) if visible.any() else np.nan,
                  np.nanpercentile(bpx[visible], 25) if visible.any() else np.nan,
                  np.mean(bpx[visible] >= 10) if visible.any() else np.nan,
                  np.nanmedian(tpx), np.nanpercentile(tpx[np.isfinite(tpx)], 25)))
        return visible.mean(), np.nanmedian(bpx[visible]) if visible.any() else np.nan

    typ = np.array([c[2] for c in cands])
    q = np.array([c[3] for c in cands])
    in_graded = np.array([graded[min(c[0], n - 1)] for c in cands])
    res = summary("all candidates", in_graded)
    summary("  insert candidates", in_graded & (typ == "ins"))
    summary("  delete candidates", in_graded & (typ == "del"))
    summary("  uncertain (0.25 <= p < 0.75)", in_graded & (q >= 0.25) & (q < 0.75))
    summary("  changed by the verifier", in_graded & (((typ == "ins") & (q >= 0.65)) | ((typ == "del") & (q < 0.35))))
    print("go / no-go (>= 0.60 visible and median ball >= ~10 px): visible %.3f, median ball %.1f px -> %s" % (
        res[0], res[1], "GO" if res[0] >= 0.6 and res[1] >= 10 else "NO-GO"))


if __name__ == "__main__":
    main()
