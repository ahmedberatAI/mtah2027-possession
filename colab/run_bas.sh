#!/bin/bash
# GPU pilot (round 7, research R-6) on a Colab VM with the user's Drive mounted (colab drivemount): the 2023
# SoccerNet ball-action-spotting winner, one fold model, on the BAR-ATH broadcast video (colab/bas_pilot.py).
# Launched in the background by colab/launch.py; log in logs/run_bas.log. Prints aggregates only.
# The video path on the mounted Drive comes from /content/bas_video.txt (one line) or $BAS_VIDEO.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
echo "start $(date -u) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) cpus $(nproc)"
pip -q install timm==0.9.2 pytorch-argus==1.0.0 kornia gdown
[ -f /content/bas_video.txt ] && export BAS_VIDEO="$(cat /content/bas_video.txt)"
python colab/bas_pilot.py
echo "done $(date -u)"
