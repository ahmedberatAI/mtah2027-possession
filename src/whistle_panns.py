"""Referee-whistle evidence from a pretrained AudioSet tagger (PANNs CNN14, Kong et al. 2020),
one value per video frame. Runs locally on CPU; nothing leaves the machine.

    .venv/Scripts/python.exe src/whistle_panns.py [sed|clip16|clip32 ...]

Variants (all write cache/whistle_panns_<variant>.npz, gitignored):
  sed     Cnn14_DecisionLevelMax (32 kHz): the model's own segment-wise output (one step = 32 mel
          hops = 0.32 s), run at 4 phase shifts of 80 ms and interleaved -> 12.5 Hz.
  clip16  Cnn14 16 kHz clip tagger: conv features of long chunks, then the clip head (max + mean
          over time, fc1, fc_audioset) on a sliding window of +-HALF_WIN feature steps; 4 phase
          shifts -> 12.5 Hz.
  clip32  the same with the 32 kHz Cnn14.

Stored per variant: `t` (step centres, seconds) and `p` (steps x 527 AudioSet probabilities,
float16); with --emb also `emb` (steps x 2048 fc1 embedding, float16). `frame_scores()`
interpolates a class onto the 199,319 video frames (video frame = round(seconds * 25)).

Weights and the AudioSet label list are fetched once into ~/panns_data (the panns_inference
convention) from Zenodo record 3987831 and the AudioSet site.
"""
import os
import subprocess
import sys
import time
import urllib.request
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
VIDEO = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical.mp4")
PANNS = os.path.join(os.path.expanduser("~"), "panns_data")
ZENODO = "https://zenodo.org/records/3987831/files/{}?download=1"
LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
N_FRAMES = 199319
FPS = 25

# AudioSet indices of the classes looked at downstream
CLASSES = {"Whistle": 402, "Whistling": 40, "Steam whistle": 403, "Train whistle": 330,
           "Crowd": 69, "Cheering": 66, "Applause": 67, "Chant": 30, "Hubbub": 70, "Speech": 0,
           "Sine wave": 501, "Beep": 481, "Squeal": 485, "Chirp": 113, "Bird": 111, "Music": 137}
VARIANTS = {
    "sed": dict(ckpt="Cnn14_DecisionLevelMax_mAP=0.385.pth", sr=32000, win=1024, hop=320, fmax=14000),
    "clip16": dict(ckpt="Cnn14_16k_mAP=0.438.pth", sr=16000, win=512, hop=160, fmax=8000),
    "clip32": dict(ckpt="Cnn14_mAP=0.431.pth", sr=32000, win=1024, hop=320, fmax=14000),
}
STEP_HOPS = 32          # CNN14 pools time by 32: one feature step = 32 mel hops = 0.32 s
SHIFTS = 4              # phase shifts per step -> 0.08 s resolution
HALF_WIN = 1            # clip head over 2*HALF_WIN+1 steps (~1 s)
CORE_STEPS = 192        # 61.44 s of output per chunk
PAD_STEPS = 16          # 5.12 s of context on each side


def _fetch(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("downloading", url)
    urllib.request.urlretrieve(url, path + ".part")
    os.replace(path + ".part", path)
    return path


def ensure_assets(variant):
    _fetch(LABELS_URL, os.path.join(PANNS, "class_labels_indices.csv"))
    ck = VARIANTS[variant]["ckpt"]
    return _fetch(ZENODO.format(ck.replace("=", "%3D")), os.path.join(PANNS, ck))


def audio_path(sr):
    """Mono PCM WAV of the tactical video's audio at `sr` (extracted once with ffmpeg)."""
    path = os.path.join(CACHE, "tactical_audio_{}k.wav".format(sr // 1000))
    if not os.path.isfile(path):
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", VIDEO,
                        "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", path], check=True)
    return path


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x, sr


def build_model(variant):
    import torch
    ensure_assets(variant)          # the label CSV must exist before panns_inference is imported
    from panns_inference.models import Cnn14, Cnn14_DecisionLevelMax
    v = VARIANTS[variant]
    cls = Cnn14_DecisionLevelMax if variant == "sed" else Cnn14
    model = cls(sample_rate=v["sr"], window_size=v["win"], hop_size=v["hop"], mel_bins=64, fmin=50,
                fmax=v["fmax"], classes_num=527)
    model.load_state_dict(load_checkpoint(os.path.join(PANNS, v["ckpt"]))["model"])
    model.eval()
    return model


def load_checkpoint(path):
    """torch.load with weights_only=True; the PANNs files (legacy format) also hold numpy arrays
    (sampler state), so only numpy's array/dtype reconstructors are allowlisted -- no arbitrary
    unpickling."""
    import numpy.dtypes as nd
    import torch
    from numpy._core import multiarray as ma
    safe = [(ma._reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray, (np.ndarray, "numpy.ndarray"),
            np.dtype, (np.dtype, "numpy.dtype")] + [getattr(nd, n) for n in dir(nd) if n.endswith("DType")]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location="cpu", weights_only=True)


def conv_steps(model, x):
    """CNN14 trunk on a batch of waveforms -> (batch, steps, 2048) after the frequency mean."""
    h = model.logmel_extractor(model.spectrogram_extractor(x))
    h = model.bn0(h.transpose(1, 3)).transpose(1, 3)
    for blk in (model.conv_block1, model.conv_block2, model.conv_block3, model.conv_block4,
                model.conv_block5):
        h = blk(h, pool_size=(2, 2), pool_type="avg")
    h = model.conv_block6(h, pool_size=(1, 1), pool_type="avg")
    return h.mean(dim=3).transpose(1, 2)


def head(model, variant, h):
    """(batch, steps, 2048) -> per-step probabilities (batch, steps, 527), fc1 embedding."""
    import torch
    import torch.nn.functional as F
    if variant == "sed":       # the model's own segment-wise output
        g = h.transpose(1, 2)
        g = (F.max_pool1d(g, 3, 1, 1) + F.avg_pool1d(g, 3, 1, 1)).transpose(1, 2)
    else:                      # clip head over a sliding window of 2*HALF_WIN+1 steps
        g = h.transpose(1, 2)
        k = 2 * HALF_WIN + 1
        g = (F.max_pool1d(g, k, 1, HALF_WIN) + F.avg_pool1d(g, k, 1, HALF_WIN,
                                                            count_include_pad=False)).transpose(1, 2)
    e = F.relu(model.fc1(g))
    return torch.sigmoid(model.fc_audioset(e)), e


def run(variant, threads=None, with_emb=False):
    import torch
    if threads:
        torch.set_num_threads(threads)
    v = VARIANTS[variant]
    model = build_model(variant)
    x, sr = read_wav(audio_path(v["sr"]))
    assert sr == v["sr"], (sr, v["sr"])
    step = STEP_HOPS * v["hop"]                    # samples per feature step (0.32 s)
    sub = step // SHIFTS                           # 0.08 s
    core, pad = CORE_STEPS * step, PAD_STEPS * step
    n_steps = int(np.ceil(len(x) / step))
    probs = np.zeros((SHIFTS, n_steps, 527), np.float16)
    embs = np.zeros((SHIFTS, n_steps, 2048), np.float16) if with_emb else None
    t0 = time.time()
    with torch.inference_mode():
        for s in range(SHIFTS):
            # shifted signal: step i of shift s covers samples [i*step + s*sub, (i+1)*step + s*sub)
            xs = np.concatenate([x[s * sub:], np.zeros(core + 2 * pad, np.float32)])
            starts = list(range(0, n_steps * step, core))
            for b0 in range(0, len(starts), 4):
                batch = []
                for c in starts[b0:b0 + 4]:
                    lo = c - pad
                    seg = xs[max(0, lo):c + core + pad]
                    if lo < 0:
                        seg = np.concatenate([np.zeros(-lo, np.float32), seg])
                    batch.append(seg)
                h = conv_steps(model, torch.from_numpy(np.stack(batch)))
                pr, e = head(model, variant, h)
                for j, c in enumerate(starts[b0:b0 + 4]):
                    i0 = c // step
                    m = min(CORE_STEPS, n_steps - i0)
                    probs[s, i0:i0 + m] = pr[j, PAD_STEPS:PAD_STEPS + m].numpy()
                    if embs is not None:
                        embs[s, i0:i0 + m] = e[j, PAD_STEPS:PAD_STEPS + m].numpy().astype(np.float16)
            print("  %s shift %d/%d done (%.0f s)" % (variant, s + 1, SHIFTS, time.time() - t0))
            sys.stdout.flush()
    # interleave shifts: output k = (step i, shift s) with k = i*SHIFTS + s, centred at
    # (i + 0.5) * 0.32 s + s * 0.08 s
    p = probs.transpose(1, 0, 2).reshape(-1, 527)
    t = ((np.arange(n_steps)[:, None] + 0.5) * step + np.arange(SHIFTS)[None, :] * sub).reshape(-1) / sr
    out = {"t": t, "p": p}
    if embs is not None:
        out["emb"] = embs.transpose(1, 0, 2).reshape(-1, 2048)
    np.savez(os.path.join(CACHE, "whistle_panns_%s.npz" % variant), **out)
    print("%s: %d steps in %.0f s" % (variant, len(t), time.time() - t0))


def load(variant):
    d = np.load(os.path.join(CACHE, "whistle_panns_%s.npz" % variant))
    return {k: d[k] for k in d.files}


def frame_scores(variant, cls="Whistle", n=N_FRAMES):
    """Probability of `cls` (a CLASSES name or an AudioSet index) interpolated onto video frames."""
    d = load(variant)
    j = CLASSES[cls] if isinstance(cls, str) else int(cls)
    return np.interp(np.arange(n) / FPS, d["t"], d["p"][:, j].astype(np.float32)).astype(np.float32)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for name in (args or ["sed", "clip16", "clip32"]):
        run(name, with_emb="--emb" in sys.argv)
