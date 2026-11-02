#!/bin/bash
# Round 22, step 1, on a Colab VM with the user's Drive mounted (colab drivemount): the SoccerNet 2025 Team-BAS baseline
# (T-DEED with a team head) on the BAR-ATH broadcast video (colab/tdeed_barath.py). The devkit and its public checkpoint
# are fetched on the VM, never through the user's connection. Launched in the background by colab/launch.py; log in
# logs/run_tdeed.log. The video path on the mounted Drive comes from /content/bas_video.txt. Prints aggregates only.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
echo "start $(date -u) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) cpus $(nproc)"
W=/content/tdeed
mkdir -p $W
[ -d $W/sn-teamspotting ] || git clone -q --depth 1 https://github.com/SoccerNet/sn-teamspotting $W/sn-teamspotting
pip -q install timm==1.0.11 SoccerNet==0.1.60 gdown moviepy==1.0.3 2>&1 | grep -v -i "warning\|notice" | tail -3 || true
if [ -z "$(find $W/ckpt -name '*.pt' 2>/dev/null)" ]; then
    gdown -q --folder https://drive.google.com/drive/folders/16IqSkctIGp76ZYKKvJvMB_ggHcQsessM -O $W/ckpt
fi
echo "checkpoint: $(find $W/ckpt -type f | wc -l) files, $(du -sh $W/ckpt | cut -f1)"
F=$W/frames/barath/broadcast
if [ ! -f $W/frames.done ]; then
    mkdir -p $F
    cp "$(cat /content/bas_video.txt)" $W/broadcast.mp4
    echo "video copied $(date -u) ($(du -h $W/broadcast.mp4 | cut -f1))"
    ffmpeg -loglevel error -threads 0 -i $W/broadcast.mp4 -vf scale=796:448 -q:v 2 -start_number 0 $F/frame%d.jpg
    touch $W/frames.done
fi
echo "frames $(find $F -name 'frame*.jpg' | wc -l) $(date -u)"
python colab/tdeed_barath.py
echo "done $(date -u)"
