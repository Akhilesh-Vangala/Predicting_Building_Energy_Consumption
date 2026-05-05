#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

LOG=results/logs_notebooks.txt
: > "$LOG"

echo "[$(date '+%H:%M:%S')] rebuilding .ipynb from .py sources" | tee -a "$LOG"
python notebooks/_build_notebooks.py 2>&1 | tee -a "$LOG"

NBCONVERT="jupyter nbconvert --to notebook --execute --inplace"

for nb in \
    notebooks/01_eda.ipynb \
    notebooks/02_feature_engineering.ipynb \
    notebooks/03_linear_models.ipynb \
    notebooks/04_tree_models.ipynb \
    notebooks/05_time_series.ipynb \
    notebooks/06_mlp.ipynb \
    notebooks/07_clustering.ipynb \
    notebooks/08_analysis.ipynb
do
    name=$(basename "$nb")
    echo "[$(date '+%H:%M:%S')] executing $name" | tee -a "$LOG"
    $NBCONVERT \
        --ExecutePreprocessor.timeout=1800 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$nb" 2>&1 | tee -a "$LOG"
    echo "[$(date '+%H:%M:%S')] done: $name" | tee -a "$LOG"
done

echo "[$(date '+%H:%M:%S')] all notebooks executed" | tee -a "$LOG"
