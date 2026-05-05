#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

CFG=configs/default.yaml
LOG=results/logs_remaining.txt
: > "$LOG"

echo "[$(date '+%H:%M:%S')] === step 1/4: raw-features ablation ===" | tee -a "$LOG"
python -m scripts.train_all --config "$CFG" --feature-set raw --skip arima lstm 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === step 2/4: arima + lstm on engineered ===" | tee -a "$LOG"
python -m scripts.train_all --config "$CFG" --feature-set engineered --only arima lstm 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === step 3/4: clustering experiment ===" | tee -a "$LOG"
python -m scripts.run_clustering --config "$CFG" 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === step 4/4: failure analysis on raw ===" | tee -a "$LOG"
python -m scripts.run_failure_analysis --config "$CFG" --feature-set raw 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === all done ===" | tee -a "$LOG"
