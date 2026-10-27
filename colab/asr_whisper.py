"""Round 11, idea B1 (docs/RAPOR.md): commentary ASR of the broadcast audio on a Colab GPU (faster-whisper,
word timestamps). Input /content/broadcast_audio.opus (uploaded; extracted locally from the broadcast video);
output /content/mtah/out/asr_words.json.gz = [[start_s, end_s, word, prob], ...] on the broadcast audio timeline.
Prints AGGREGATES only (language, counts, timing) -- never the transcript.
"""
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import squads  # noqa: E402

AUDIO = os.environ.get("ASR_AUDIO", "/content/broadcast_audio.opus")
MODEL = os.environ.get("ASR_MODEL", "large-v3")
OUT = "/content/mtah/out/asr_words.json.gz"


def load_audio(path, sr=16000):
    """Mono float32 at 16 kHz via the ffmpeg binary (faster-whisper's PyAV decoder breaks on some av versions)."""
    import subprocess
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr),
                          "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def main():
    from faster_whisper import WhisperModel
    t0 = time.time()
    model = WhisperModel(MODEL, device="cuda", compute_type="float16")
    print("model %s loaded in %.0f s" % (MODEL, time.time() - t0))
    audio = load_audio(AUDIO)
    segments, info = model.transcribe(audio, language=None, beam_size=5, word_timestamps=True, vad_filter=True,
                                      condition_on_previous_text=False, hotwords=squads.prompt_names())
    print("language %s p=%.2f duration %.0f s" % (info.language, info.language_probability, info.duration))
    words, n_seg, last = [], 0, time.time()
    for seg in segments:
        n_seg += 1
        for w in seg.words or []:
            words.append([round(w.start, 3), round(w.end, 3), w.word, round(w.probability, 3)])
        if time.time() - last > 120:
            print("  %.0f / %.0f s of audio, %d words (%.0f s)" % (seg.end, info.duration, len(words), time.time() - t0))
            sys.stdout.flush()
            last = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        json.dump(words, fh)
    probs = sorted(w[3] for w in words)
    ment = squads.spot([(w[0], w[1], w[2]) for w in words])
    pl = [m for m in ment if m[3] == "player"]
    print("segments %d words %d (%.0f / min of audio) median word prob %.2f; %.0f s total" % (
        n_seg, len(words), 60 * len(words) / max(1, info.duration), probs[len(probs) // 2] if probs else 0,
        time.time() - t0))
    print("name mentions: players home %d away %d, team names home %d away %d, distinct players %d" % (
        sum(m[2] == 0 for m in pl), sum(m[2] == 1 for m in pl),
        sum(m[2] == 0 and m[3] == "team" for m in ment), sum(m[2] == 1 and m[3] == "team" for m in ment),
        len({m[4] for m in pl})))


if __name__ == "__main__":
    main()
