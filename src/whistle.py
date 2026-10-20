"""Referee-whistle evidence from the tactical video's audio track, one value per video frame.

STFT with a 40 ms hop (= one video frame at 25 fps). A whistle is a loud, narrow-band tone in
roughly 2.5-4.6 kHz: per frame we take the band's spectral peak minus the band median (tonality)
and subtract a slow rolling median of that score (crowd / ambience baseline).
"""
import os
import wave

import numpy as np
from scipy.signal import stft

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV = os.path.join(ROOT, "cache", "tactical_audio_16k.wav")
OUT = os.path.join(ROOT, "cache", "whistle_score.npz")


def read_wav(path=WAV):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x, sr


def rolling_median(x, half):
    """Approximate rolling median via strided percentile on blocks (fast enough here)."""
    n = len(x)
    out = np.empty(n, np.float32)
    step = max(1, half // 4)
    centers = np.arange(0, n, step)
    vals = np.array([np.median(x[max(0, c - half):min(n, c + half + 1)]) for c in centers])
    out[:] = np.interp(np.arange(n), centers, vals)
    return out


def compute(n_frames=199319, band=(2500, 4600), refresh=False):
    if os.path.isfile(OUT) and not refresh:
        d = np.load(OUT)
        return {k: d[k] for k in d.files}
    x, sr = read_wav()
    hop = sr // 25                      # 640 samples = 40 ms
    f, t, Z = stft(x, fs=sr, nperseg=1024, noverlap=1024 - hop, boundary=None, padded=False)
    mag = 20 * np.log10(np.abs(Z) + 1e-8).astype(np.float32)   # (freq, time)
    b = (f >= band[0]) & (f <= band[1])
    lowband = (f >= 300) & (f <= 2000)
    peak = mag[b].max(0)
    med = np.median(mag[b], axis=0)
    tonal = peak - med                                          # dB above the band floor
    band_e = 10 * np.log10(np.mean(10 ** (mag[b] / 10), axis=0))
    low_e = 10 * np.log10(np.mean(10 ** (mag[lowband] / 10), axis=0))
    peak_f = f[b][mag[b].argmax(0)]
    # frame k of the STFT is centred at (k*hop + 512)/sr seconds -> video frame
    centres = (np.arange(mag.shape[1]) * hop + 512) / sr
    vf = np.round(centres * 25).astype(int)
    def to_frames(v):
        out = np.full(n_frames, np.nan, np.float32)
        ok = (vf >= 0) & (vf < n_frames)
        out[vf[ok]] = v[ok]
        # fill any holes
        idx = np.flatnonzero(np.isfinite(out))
        return np.interp(np.arange(n_frames), idx, out[idx]).astype(np.float32)
    tonal_f = to_frames(tonal)
    ratio_f = to_frames(band_e - low_e)
    base_t = rolling_median(tonal_f, 750)       # +-30 s baseline
    base_r = rolling_median(ratio_f, 750)
    res = {"tonal": tonal_f, "tonal_ex": tonal_f - base_t, "ratio_ex": ratio_f - base_r,
           "peak_hz": to_frames(peak_f)}
    np.savez(OUT, **res)
    return res
