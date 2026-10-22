#!/bin/bash
# GPU job for idea N25 on a Colab VM (launched in the background by colab/launch.py; log in logs/run_gru.log).
# Fetches open data on the VM, builds the sim-V3 bench inputs / v7 features, trains and compares the BiGRU and
# LightGBM team models (src/exp_gru.py). Prints aggregates only.
set -e
cd "$(dirname "$0")/.."
export BENCH_THREADS=$(nproc) BENCH_PROCS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 )) PYTHONUNBUFFERED=1
echo "start $(date -u) cpus $(nproc)"
python colab/fetch_open_data.py "$PWD"
python -c "import sys; sys.path.insert(0, 'src'); import bench; [bench.blocks(s) for s in ('dfl@v3', 'metrica@v3', 'sc@v3')]; print('inputs ready')"
python src/exp_gru.py
echo "done $(date -u)"
