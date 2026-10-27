"""Round 11 diagnostic: is there commentary (speech) in the broadcast audio? PANNs CNN14 16 kHz clip tagger
(src/whistle_panns.py, local weights) on the broadcast and the tactical audio. AGGREGATE output only (class
shares); the per-step probabilities stay in cache/ (frame-level derivative, gitignored).

    python -I src/exp_audio.py
"""
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whistle_panns as wp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
BROADCAST = os.path.join(ROOT, "data", "broadcast", "BAR-ATH_broadcast.mp4")
# AudioSet indices: speech family, crowd family, music
SHOW = {"Speech": 0, "Male speech": 1, "Female speech": 2, "Conversation": 4, "Narration": 5,
        "Crowd": 69, "Cheering": 66, "Chant": 30, "Music": 137, "Whistle": 402}
# graded halves on each timeline (broadcast = tactical - offset; offsets from src/broadcast.py)
TAC_HALVES = [(11185, 90651), (110808, 190197)]
OFFSETS = [4014, 26920]


def broadcast_wav(sr=16000):
    path = os.path.join(CACHE, "broadcast_audio_16k.wav")
    if not os.path.isfile(path):
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", BROADCAST,
                        "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", path], check=True)
    return path


def tag(path, out):
    """One phase shift of the clip16 tagger over the whole file -> (t seconds, probs float16)."""
    if os.path.isfile(out):
        d = np.load(out)
        return d["t"], d["p"]
    import torch
    variant = "clip16"
    v = wp.VARIANTS[variant]
    model = wp.build_model(variant)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    x, sr = wp.read_wav(path)
    step = wp.STEP_HOPS * v["hop"]
    core, pad = wp.CORE_STEPS * step, wp.PAD_STEPS * step
    n_steps = int(np.ceil(len(x) / step))
    probs = np.zeros((n_steps, 527), np.float16)
    xs = np.concatenate([x, np.zeros(core + 2 * pad, np.float32)])
    t0 = time.time()
    with torch.inference_mode():
        starts = list(range(0, n_steps * step, core))
        for b0 in range(0, len(starts), 8):
            batch = []
            for c in starts[b0:b0 + 8]:
                lo = c - pad
                seg = xs[max(0, lo):c + core + pad]
                if lo < 0:
                    seg = np.concatenate([np.zeros(-lo, np.float32), seg])
                batch.append(seg)
            h = wp.conv_steps(model, torch.from_numpy(np.stack(batch)).to(dev))
            pr, _ = wp.head(model, variant, h)
            pr = pr.float().cpu().numpy()
            for j, c in enumerate(starts[b0:b0 + 8]):
                i0 = c // step
                m = min(wp.CORE_STEPS, n_steps - i0)
                probs[i0:i0 + m] = pr[j, wp.PAD_STEPS:wp.PAD_STEPS + m]
    t = (np.arange(n_steps) + 0.5) * step / sr
    np.savez(out, t=t, p=probs)
    print("  tagged %s: %d steps in %.0f s" % (os.path.basename(path), n_steps, time.time() - t0))
    return t, probs


def summary(name, t, p, halves):
    """Share of 0.32 s steps (inside the graded halves) where each class has probability > 0.5, and its mean."""
    m = np.zeros(len(t), bool)
    for a, b in halves:
        m |= (t >= a / 25.0) & (t <= b / 25.0)
    q = p[m].astype(np.float32)
    parts = ["%s %.2f/%.2f" % (k, (q[:, j] > 0.5).mean(), q[:, j].mean()) for k, j in SHOW.items()]
    print("%-10s (%.0f min graded) share>0.5 / mean: %s" % (name, m.sum() * 0.32 / 60, " | ".join(parts)))
    top = np.argsort(-q.mean(0))[:8]
    print("           top classes by mean:", ", ".join("%d:%.2f" % (j, q[:, j].mean()) for j in top))


if __name__ == "__main__":
    bt, bp = tag(broadcast_wav(), os.path.join(CACHE, "audio_panns_broadcast_clip16.npz"))
    tt, tp = tag(wp.audio_path(16000), os.path.join(CACHE, "audio_panns_tactical_clip16.npz"))
    summary("broadcast", bt, bp, [(a - o, b - o) for (a, b), o in zip(TAC_HALVES, OFFSETS)])
    summary("tactical", tt, tp, TAC_HALVES)
    sys.stdout.flush()
