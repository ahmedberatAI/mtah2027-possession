#!/bin/bash
# Round 23 GPU job (launched in the background by colab/launch.py; log in logs/run_kpu.log): the short-possession
# specialist (src/short_specialist.py, docs/RAPOR.md Tur 23). PFF is copied from the provider's folder in the user's
# mounted Drive (as colab/run_bigru_lomo.sh; colab/fetch_pff.py is the fallback), parsed (src/gradient.py), degraded
# twice (sim.V3, sim.V4) and turned into features, targets and v10's prior; then two variants x three seeds are trained
# on the GPU. Models and training logs go to /content/mtah/out/kpu and the user's Drive (MyDrive/mtah_out/kpu). No
# BAR-ATH, DFL, Metrica or SkillCorner data on the VM. Prints aggregates only.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 BENCH_THREADS=2
echo "start $(date -u) cpus $(nproc) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
pip -q install orjson lightgbm==4.7.0 numba 2>&1 | grep -v -i "warning\|notice" | tail -2 || true
if [ -z "$PFF_DIR" ] && [ -d /content/drive ]; then
    PFF_DIR=$(find -L /content/drive -maxdepth 4 -type d -name "Tracking Data" 2>/dev/null | while read d; do
        p=$(dirname "$d"); [ -d "$p/Event Data" ] && [ -d "$p/Rosters" ] && echo "$p" && break; done)
fi
if [ -n "$PFF_DIR" ]; then
    echo "PFF from the mounted Drive"
    for pair in "events:Event Data" "metadata:Metadata" "rosters:Rosters" "tracking:Tracking Data"; do
        kind=${pair%%:*}; name=${pair#*:}
        mkdir -p external_data/gradient/$kind
        find "$PFF_DIR/$name" -maxdepth 1 -type f -name '[0-9]*.json*' -exec cp -n {} external_data/gradient/$kind/ \;
        echo "pff $kind $(ls external_data/gradient/$kind | wc -l) files"
    done
else
    python colab/fetch_pff.py "$PWD"
fi
# bench (imported for v10's prior) imports src/evaluate.py, which needs the official scorer
SCORER=external/more-than-a-hack-2027/evaluation/possession_metrics.py
if [ ! -f $SCORER ]; then
    mkdir -p $(dirname $SCORER)
    curl -sfL -o $SCORER https://raw.githubusercontent.com/metrica-sports/more-than-a-hack-2027/f8f2629ff1e0985f81d7e59d8413678514ee1530/evaluation/possession_metrics.py
fi
D=/content/kpu_data
python -I src/short_specialist.py prep-pff $D $(nproc)
echo "prepared $(date -u) ($(du -sh $D | cut -f1))"
mkdir -p logs /content/mtah/out/kpu
for v in S1 S2; do
    for s in 20261102 20261103 20261104; do
        python -I src/short_specialist.py train $D $v $s /content/mtah/out/kpu > logs/kpu_${v}_$s.log 2>&1 &
    done
    wait
    grep -h "step 12000\|Traceback\|Error" logs/kpu_${v}_*.log | head -6
done
if [ -d /content/drive/MyDrive ]; then
    mkdir -p /content/drive/MyDrive/mtah_out && cp -r /content/mtah/out/kpu /content/drive/MyDrive/mtah_out/
    echo "copied to the user's Drive: MyDrive/mtah_out/kpu"
fi
echo "done $(date -u)"
