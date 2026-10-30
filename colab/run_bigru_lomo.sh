#!/bin/bash
# Round 17 GPU job (launched in the background by colab/launch.py; log in logs/run_bigru_lomo.log): leave-one-DFL-
# match-out training of the shared BiGRU recipe (src/exp_bigru_lomo.py). Open DFL data, the Metrica sample games and
# the official scorer are fetched on the VM. The PFF
# World Cup data is copied from the provider's folder in the user's mounted Drive (a shortcut to it in My Drive,
# `colab drivemount` authorised by the user; PFF_DIR overrides the search); anonymous downloads of it hit Google's
# quota (colab/fetch_pff.py is the fallback). No BAR-ATH data. Prints aggregates only.
set -e
cd "$(dirname "$0")/.."
export BENCH_THREADS=$(nproc) BENCH_PROCS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 )) PYTHONUNBUFFERED=1
echo "start $(date -u) cpus $(nproc) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
pip -q install orjson lightgbm==4.7.0 numba gdown
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
# src/evaluate.py (imported through bench) needs the official scorer; bench builds the DFL and Metrica sample inputs
# together, so both open sets are fetched (SkillCorner is not needed here)
SCORER=external/more-than-a-hack-2027/evaluation/possession_metrics.py
if [ ! -f $SCORER ]; then
    mkdir -p $(dirname $SCORER)
    curl -sfL -o $SCORER https://raw.githubusercontent.com/metrica-sports/more-than-a-hack-2027/f8f2629ff1e0985f81d7e59d8413678514ee1530/evaluation/possession_metrics.py
fi
python src/download_idsse.py
python -c "import sys; sys.path.insert(0, 'src'); import metrica_sample; metrica_sample.download()"
python src/exp_bigru_lomo.py prepare 4
mkdir -p logs
for s in 20261025 20261026 20261027; do
    python src/exp_bigru_lomo.py train $s > logs/lomo_$s.log 2>&1 &
done
wait
grep -h "held-out" logs/lomo_*.log || true
python src/exp_bigru_lomo.py collect
echo "done $(date -u)"
