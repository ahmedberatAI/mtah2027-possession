#!/bin/bash
# Round 24: variant C1 of the short-possession specialist (controlled-possession reading of PFF + touch head; no v10
# prior) through colab/run_kpu.sh. Launched by colab/launch.py; log in logs/run_kpu_c1.log.
cd "$(dirname "$0")/.."
KPU_CONV=ctrl KPU_VARIANTS=C1 KPU_DATA=/content/kpu_data_c1 bash colab/run_kpu.sh
