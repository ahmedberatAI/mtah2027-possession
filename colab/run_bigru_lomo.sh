#!/bin/bash
# Round 17 GPU job (launched in the background by colab/launch.py; log in logs/run_bigru_lomo.log): leave-one-DFL-
# match-out training of the shared BiGRU recipe (src/exp_bigru_lomo.py). Open DFL data is fetched on the VM; the PFF
# folder comes from the user's Drive, mounted at /content/drive by `colab drivemount` (the user authorises it). No
# BAR-ATH data. Prints aggregates only.
set -e
cd "$(dirname "$0")/.."
export BENCH_THREADS=$(nproc) BENCH_PROCS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 )) PYTHONUNBUFFERED=1
echo "start $(date -u) cpus $(nproc) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
pip -q install orjson lightgbm==4.7.0 numba
PFF_DIR=${PFF_DIR:-$(find /content/drive -maxdepth 7 -type d -name gradient 2>/dev/null | while read d; do
    [ -d "$d/events" ] && [ -d "$d/tracking" ] && echo "$d" && break; done)}
if [ -z "$PFF_DIR" ]; then echo "PFF folder (gradient/{events,metadata,rosters,tracking}) not found under /content/drive"; exit 1; fi
echo "PFF: $(ls "$PFF_DIR/events" | wc -l) event files, $(ls "$PFF_DIR/tracking" | wc -l) tracking files"
mkdir -p external_data && ln -sfn "$PFF_DIR" external_data/gradient
python src/download_idsse.py
python src/exp_bigru_lomo.py prepare 4
mkdir -p logs
for s in 20261025 20261026 20261027; do
    python src/exp_bigru_lomo.py train $s > logs/lomo_$s.log 2>&1 &
done
wait
grep -h "held-out" logs/lomo_*.log || true
python src/exp_bigru_lomo.py collect
echo "done $(date -u)"
