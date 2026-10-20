"""Referee-whistle detector by example (query by example on the tactical video's audio).

The referee blows the same whistle all match, so its tonal fingerprint (fundamental + harmonics)
is constant, while crowd whistling lands on arbitrary pitches. The only whistles whose timing is
known without any labelling are the two half-opening kick-off whistles (~1.1 s before the first
touch at video frames 11185 and 113742), and they are the examples.

  P[t, f]   "tonality": STFT dB (n_fft 2048 @ 32 kHz, one frame per video frame) on a log-frequency
            axis (1 % steps, 1.2-12 kHz) minus its running median across +-12 % of frequency,
            clipped at 0 -- narrow-band peaks only, broadband crowd noise removed
  template  mean P over the most tonal 0.4 s in [-2.5, -0.3] s before a kick-off minus mean P over
            [-8, -3] s (local ambience), clipped at 0
  score     max over pitch shifts (+-6 %) of the cosine between P[t] and the template, times the
            template-weighted tonality (so faint matches stay low)

Arrays in the cache: s1 / s2 (template of one kick-off -- the other kick-off is the held-out test),
s (mean of both; used by model_v6), r30 / r60 (self-trained: mean template of the top-k detections
of s outside the kick-offs), cos_* (cosine part only), tonal_max (max tonality 1.5-6 kHz).

    .venv/Scripts/python.exe src/whistle_ref.py    -> cache/whistle_ref.npz (gitignored)
Prints only aggregate numbers (template fundamental, leave-one-kickoff-out ranks).
"""
import os
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import stft

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import atd  # noqa: E402
import whistle_panns as wp  # noqa: E402

OUT = os.path.join(ROOT, "cache", "whistle_ref.npz")
P_CACHE = os.path.join(ROOT, "cache", "whistle_ref_P.npy")
N = atd.N_FRAMES["tactical"]
FPS = atd.FPS
KICKOFFS = (atd.HALVES[0][1], atd.KICKOFF_2ND)
SR = 32000
NFFT = 2048
HOP = SR // FPS                       # 1280 samples = one video frame
LOGF = 1.2e3 * 1.01 ** np.arange(232)  # 1.2 kHz .. ~12 kHz, 1 % steps
MED_BINS = 25                          # running median over +-12 bins (~ +-12 %)
SHIFTS = range(-6, 7)                  # pitch shifts in 1 % steps
REFINE_K = (30, 60)                    # self-training: templates from the top-k detections


def tonality(chunk_s=600):
    """P[t, f] for every video frame (float16), computed in chunks."""
    x, sr = wp.read_wav(wp.audio_path(SR))
    assert sr == SR
    f_lin = np.fft.rfftfreq(NFFT, 1 / SR)
    # log bin i takes the max over the linear bins inside [f_i / 1.005, f_i * 1.005) (narrow
    # peaks survive), or the nearest linear bin where that range holds none
    ranges = []
    for fc in LOGF:
        lo_b, hi_b = np.searchsorted(f_lin, fc / 1.005), np.searchsorted(f_lin, fc * 1.005)
        if hi_b <= lo_b:
            lo_b = int(np.argmin(np.abs(f_lin - fc)))
            hi_b = lo_b + 1
        ranges.append((lo_b, hi_b))
    P = np.zeros((N, len(LOGF)), np.float16)
    step = chunk_s * FPS
    for f0 in range(0, N, step):
        f1 = min(N, f0 + step)
        lo = f0 * HOP - NFFT // 2
        seg = x[max(0, lo):f1 * HOP + NFFT // 2]
        if lo < 0:
            seg = np.concatenate([np.zeros(-lo, np.float32), seg])
        seg = np.concatenate([seg, np.zeros(max(0, (f1 - f0 - 1) * HOP + NFFT - len(seg)), np.float32)])
        _, _, Z = stft(seg, fs=SR, window="hann", nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
        db = 20 * np.log10(np.abs(Z[:, :f1 - f0]) + 1e-7)          # (freq, frames); frame j centred at (f0+j)/25 s
        lg = np.stack([db[a:b].max(0) for a, b in ranges]).astype(np.float32)
        base = median_filter(lg, size=(MED_BINS, 1), mode="nearest")
        P[f0:f1] = np.clip(lg - base, 0, None).T.astype(np.float16)
    return P


def template(P, k):
    """Whistle profile before kick-off k: most tonal 0.4 s in [-2.5, -0.3] s minus [-8, -3] s."""
    w = P[k - int(2.5 * FPS):k - int(0.3 * FPS)].astype(np.float32)
    band = (LOGF >= 1.5e3) & (LOGF <= 6e3)
    strength = w[:, band].max(1)
    top = np.argsort(-strength)[:10]
    bg = P[k - 8 * FPS:k - 3 * FPS].astype(np.float32).mean(0)
    return np.clip(w[top].mean(0) - bg, 0, None), k - int(2.5 * FPS) + top


def match(P, T, block=20000):
    """max over pitch shifts of cos(P[t], T_shift) * (P[t] . T_shift / |T|)."""
    out = np.zeros(len(P), np.float32)
    cosm = np.zeros(len(P), np.float32)
    for a in range(0, len(P), block):
        X = P[a:a + block].astype(np.float32)
        nx = np.linalg.norm(X, axis=1) + 1e-6
        best = np.full(len(X), -np.inf, np.float32)
        bestc = np.zeros(len(X), np.float32)
        for s in SHIFTS:
            Ts = np.roll(T, s)
            if s > 0:
                Ts[:s] = 0
            elif s < 0:
                Ts[s:] = 0
            nt = np.linalg.norm(Ts) + 1e-6
            proj = X @ Ts / nt
            cos = proj / nx
            sc = cos * np.clip(proj, 0, None)
            upd = sc > best
            best[upd] = sc[upd]
            bestc[upd] = cos[upd]
        out[a:a + block] = best
        cosm[a:a + block] = bestc
    return out, cosm


def fundamental(T):
    band = (LOGF >= 1.5e3) & (LOGF <= 6e3)
    return float(LOGF[band][np.argmax(T[band])])


def tonality_cached(refresh=False):
    if os.path.isfile(P_CACHE) and not refresh:
        return np.load(P_CACHE)
    P = tonality()
    np.save(P_CACHE, P)
    return P


def refine(P, s, k):
    """Self-training: mean template of the k strongest peaks of s outside +-15 s of the kick-offs
    (which stay held out as the test); each example = its most tonal 3 frames within +-0.12 s
    minus the [-5, -2] s ambience."""
    import whistle_eval as we
    g = atd.graded_mask(N).copy()
    for k0 in KICKOFFS:
        g[k0 - 15 * FPS:k0 + 15 * FPS] = False
    acc = np.zeros(P.shape[1])
    pk = we.peaks(s, g)[:k]
    for p in pk:
        w = P[p - 3:p + 4].astype(np.float32)
        t = np.clip(w[np.argsort(-w.max(1))[:3]].mean(0) - P[p - 5 * FPS:p - 2 * FPS].astype(np.float32).mean(0), 0, None)
        acc += t / (np.linalg.norm(t) + 1e-6)
    return acc / len(pk)


def build(refresh=False):
    if os.path.isfile(OUT) and not refresh:
        d = np.load(OUT)
        return {k: d[k] for k in d.files}
    P = tonality_cached(refresh)
    T1, _ = template(P, KICKOFFS[0])
    T2, _ = template(P, KICKOFFS[1])
    T = (T1 / (np.linalg.norm(T1) + 1e-6) + T2 / (np.linalg.norm(T2) + 1e-6)) / 2
    res = {"T1": T1, "T2": T2, "T": T}
    for name, tpl in (("s1", T1), ("s2", T2), ("s", T)):
        res[name], res["cos_" + name] = match(P, tpl)
    for k in REFINE_K:
        res["T_r%d" % k] = refine(P, res["s"], k)
        res["r%d" % k], res["cos_r%d" % k] = match(P, res["T_r%d" % k])
    res["tonal_max"] = P[:, (LOGF >= 1.5e3) & (LOGF <= 6e3)].astype(np.float32).max(1)
    np.savez(OUT, **res)
    return res


if __name__ == "__main__":
    import whistle_eval as we
    r = build(refresh="--refresh" in sys.argv)
    f1, f2 = fundamental(r["T1"]), fundamental(r["T2"])
    c12 = float(r["T1"] @ r["T2"] / (np.linalg.norm(r["T1"]) * np.linalg.norm(r["T2"]) + 1e-9))
    print("template peak frequency: kickoff1 %.0f Hz, kickoff2 %.0f Hz; template cosine %.2f" % (f1, f2, c12))
    # leave-one-kickoff-out: template from one kick-off, rank of the other kick-off's whistle
    g = atd.graded_mask(N)
    tests = [("s1", KICKOFFS[1]), ("s2", KICKOFFS[0])]
    tests += [("r%d" % k, k0) for k in REFINE_K for k0 in KICKOFFS]
    for k in REFINE_K:
        T = r["T_r%d" % k]
        print("refined template r%d: peak frequency %.0f Hz, cosine to kick-off templates %.2f / %.2f" % (
            k, fundamental(T), T @ r["T1"] / (np.linalg.norm(T) * np.linalg.norm(r["T1"]) + 1e-9),
            T @ r["T2"] / (np.linalg.norm(T) * np.linalg.norm(r["T2"]) + 1e-9)))
    for name, k in tests:
        s = r[name]
        pk = we.peaks(s, g)
        j = k - 75 + int(np.argmax(s[k - 75:k + 6]))
        print("template %s -> held-out kick-off %d: peak %+.2f s, pct %.2f, rank %d of %d peaks" % (
            name, k, (j - k) / FPS, 100 * (s[g] < s[j]).mean(), 1 + int((s[pk] > s[j]).sum()), len(pk)))
