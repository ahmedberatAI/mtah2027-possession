#!/bin/bash
# Round 11, idea B1: commentary ASR of the broadcast audio (colab/asr_whisper.py) on a Colab GPU. The audio
# (/content/broadcast_audio.opus, ~25 MB) is uploaded with `colab upload`; the model weights are fetched ON the VM.
# Launched in the background by colab/launch.py; log in logs/run_asr.log. Prints aggregates only.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
echo "start $(date -u) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) cpus $(nproc)"
# ctranslate2 (faster-whisper) is built against CUDA 12: cuBLAS 12 + cuDNN 9 from the NVIDIA wheels
pip -q install faster-whisper nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
LIBS="$(python -c 'import nvidia.cublas.lib as a, nvidia.cudnn.lib as b; print(a.__path__[0] + ":" + b.__path__[0])')"
export LD_LIBRARY_PATH="$LIBS:$LD_LIBRARY_PATH"
echo "cuda libs $LIBS"
python colab/asr_whisper.py
echo "done $(date -u)"
