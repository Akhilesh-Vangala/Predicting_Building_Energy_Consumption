#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

CFG=configs/default.yaml
LOG=results/logs_finalize.txt
: > "$LOG"

echo "[$(date '+%H:%M:%S')] === step 1/3: arima only ===" | tee -a "$LOG"
python -m scripts.train_all --config "$CFG" --feature-set engineered --only arima 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === step 2/3: lstm only ===" | tee -a "$LOG"
python -m scripts.train_all --config "$CFG" --feature-set engineered --only lstm 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === step 3/3: clustering (with fix) ===" | tee -a "$LOG"
python -m scripts.run_clustering --config "$CFG" 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === finalize done ===" | tee -a "$LOG"
