# HANDOFF — Predicting Building Energy Consumption

This document is the single source of truth for the current state of the
project. Read it end-to-end before touching anything.

---

## 0. Identity and stakes

- **Owner**: Akhilesh Vangala (sole author).
- **Course**: NYU DS-GA-1003 Applied ML, Spring 2026, **Track 1 (Applied ML)**.
- **Final paper due**: May 8th, 11:59 PM ET (NeurIPS-style, up to 8 pages).
- **GitHub**: https://github.com/Akhilesh-Vangala/Predicting-Building-Energy-Consumption
  (the professor must be able to clone and reproduce results from this repo).
- **Hard rules — non-negotiable**:
  1. The GitHub contributor list must show **only Akhilesh**. No "Claude",
     "Cursor", "co-authored-by" trailers anywhere. Do **not** edit `git config`.
     Verify identity before any commit (`git config user.name` →
     `Akhilesh-Vangala`, email `akhileshvangala98@gmail.com`).
  2. **Code must read like a human wrote it.** No "AI giveaway" comments
     (no `# Import the module`, `# Define the function`, `# Increment counter`,
     no narrating-the-obvious). Comments only when they explain *why* something
     non-obvious was done.
  3. No emojis anywhere — code, commits, messages, paper.
  4. No checked-in secrets. The `.gitignore` already excludes `.venv`,
     `__pycache__`, prediction `.npy`s, raw data, and the proposal PDF.

---

## 1. Problem framing (from the proposal)

Predict hourly **building energy consumption** (kWh) using the **ASHRAE Great
Energy Predictor III** dataset (Kaggle, ~20M rows, 1,449 buildings, 16 sites,
4 meter types: electricity, chilledwater, steam, hotwater). Final paper must:

- Compare **10 ML models** across 4 families:
  - Linear: OLS, Ridge, Lasso, ElasticNet
  - Trees / boosting: Decision Tree, Random Forest, LightGBM
  - Time-series: ARIMA, LSTM
  - Neural (tabular): MLP
- Use **time-based split**: train Jan–Sep 2016, validate Oct–Dec 2016.
- Use **~28 engineered features** in 7 groups (see `src/features/pipeline.py`):
  - time (8): hour_sin/cos, dayofweek_sin/cos, month_sin/cos, is_weekend, is_holiday
  - lag (3): lag_24h, lag_168h, lag_diff_24h
  - rolling (4): rolling_mean/std for 24h and 168h windows
  - weather (6): air_temperature, dew_temperature, temp_squared, wind_speed,
    cloud_coverage, precip_depth_1_hr
  - building (3): log_square_feet, building_age, primary_use_code
  - interaction (2): hour_x_weekend, sqft_x_temp
  - target encoding (1): te_primary_use_x_hour (smoothing=30)
- Run **4 explicit analyses** the proposal commits to:
  1. **Feature-engineering ablation**: 5-feature raw set vs full 28-feature set.
  2. **Temporal modeling value**: ARIMA / LSTM vs static-feature models.
  3. **Per-building / meter-type failure modes**: where each model collapses.
  4. **Clustering benefit**: K-means on hourly profiles, per-cluster LightGBM
     vs single global LightGBM.
- Subset for compute reasons: **sites [2, 4, 13, 14]** (482 buildings,
  ~8.5M rows pre-clean, ~7.9M post-clean). The full 16-site run is optional
  if compute permits.
- Primary metric **RMSE (kWh)**, secondaries: MAE, CV-RMSE, RMSLE, all with
  per-primary_use / per-meter / per-site breakdowns.

The proposal PDF lives at `report/proposal.pdf` (also `ml_proposal.pdf` in
the workspace root, gitignored).

---

## 2. Repo layout (everything already in place)

```
configs/
  default.yaml      # 4-site subset, full hyperparam config
  smoke.yaml        # 5-meter smoke test
data/
  raw/              # ASHRAE CSVs (gitignored). User keeps them in
                    # ~/Desktop/metrix ai/ — symlink or copy as needed
  processed/        # Feature-engineered parquet (optional cache)
src/
  config.py         # YAML -> dataclass loader
  utils.py          # logging, timer, save_json, set_seed
  pipeline.py       # prepare_data(cfg, feature_set), evaluate_predictions,
                    # baselines (meter_mean, lag_24h)
  data/
    load.py         # load_subset, merge_with_context (kBTU->kWh for site 0)
    clean.py        # outlier cap, zero-streak removal, weather imputation
    split.py        # time_split by month
  features/
    pipeline.py     # FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, build_features
    time_features.py / lag_features.py / weather_features.py /
    building_features.py / interactions.py / target_encoding.py
  models/
    base.py         # BaseModel ABC with feature_importance()
    linear.py       # OLS/Ridge/Lasso/ElasticNet, _LinearScaledMixin filters
                    # zero-variance cols before StandardScaler
    trees.py        # DecisionTree, RandomForest (uses max_samples for speed)
    boosting.py     # LightGBM (early stopping, eval_set)
    arima.py        # auto_arima per meter, fallback to seasonal-naive
    lstm.py         # 168h lookback, single-layer (config-driven)
    mlp.py          # 3-layer with dropout + early stopping
  evaluation/
    metrics.py      # rmse, mae, cv_rmse, rmsle, score_predictions
    per_group.py    # per_primary_use / per_meter / per_site breakdowns
  clustering/
    kmeans_profiles.py  # build_consumption_profiles, fit_kmeans, elbow_curve
  viz/
    eda_plots.py / result_plots.py / style.py
scripts/
  download_data.py        # one-shot Kaggle fetch
  build_features.py       # standalone feature build
  train_all.py            # main entry; --only / --skip / --refit
  run_ablation.py         # raw vs engineered ablation
  run_clustering.py       # K-means + per-cluster LightGBM vs global
  run_failure_analysis.py # per-group breakdown tables
  run_remaining.sh        # historical chain (raw ablation + arima/lstm + cluster + failure)
  run_finalize.sh         # ARIMA + LSTM + clustering only
notebooks/
  01_eda.ipynb            # EDA: distributions, correlations, missingness, temporal
  02_feature_engineering.ipynb
  03_linear_models.ipynb
  04_tree_models.ipynb
  05_time_series.ipynb
  06_mlp.ipynb
  07_clustering.ipynb
  08_analysis.ipynb       # ablation, failure, gap analysis
  _build_notebooks.py     # generates .ipynb from sibling .py files (jupytext-style)
tests/
  test_clean.py / test_features.py / test_metrics.py / test_split.py
report/
  proposal.pdf
results/                  # generated artifacts (CSVs and JSON committed,
                          # large .npy predictions gitignored)
Makefile                  # `make data | features | train | report | all`
requirements.txt          # pinned deps incl. lightgbm, pmdarima, torch, pytest
README.md                 # repro instructions
```

---

## 3. What is already done (verified working)

### 3.1 Engineered-feature run (`results/tables/models_engineered.csv`)
All 8 trained models + 2 baselines with per-primary_use / per-meter / per-site
breakdowns saved in `results/metrics/models_engineered.json`.

| Model | Family | RMSE (kWh) | MAE | CV-RMSE | RMSLE |
|-------|--------|-----------:|----:|--------:|------:|
| random_forest | trees | **4,314** | 146 | 5.72 | 0.69 |
| decision_tree | trees | 4,991 | 176 | 6.61 | 0.74 |
| lightgbm | trees | 5,731 | 200 | 7.59 | 0.68 |
| elasticnet | linear | 6,319 | 675 | 8.37 | 1.75 |
| ols | linear | 6,428 | 677 | 8.52 | 1.73 |
| ridge | linear | 6,428 | 677 | 8.52 | 1.73 |
| lasso | linear | 6,558 | 677 | 8.69 | 1.76 |
| mlp | neural | 10,085 | 332 | 13.36 | 0.90 |
| baseline_lag_24h | baseline | 83,959 | 614 | 76.40 | 0.91 |
| baseline_meter_mean | baseline | 175,277 | 6,142 | 159.49 | 1.46 |

### 3.2 Raw-feature ablation (`results/tables/models_raw.csv`)
8 models (no ARIMA/LSTM) on the 5-feature raw set. Provides the
feature-engineering ablation table.

| Model | engineered | raw | Δ |
|-------|-----------:|----:|--:|
| ridge | 6,428 | 7,033 | +9% |
| lasso | 6,558 | 7,046 | +7% |
| elasticnet | 6,319 | 7,046 | +12% |
| decision_tree | 4,991 | 30,845 | +518% |
| random_forest | 4,314 | 30,834 | +615% |
| lightgbm | 5,731 | 28,951 | +405% |
| mlp | 10,085 | 11,252 | +12% |

Story: **trees collapse without engineered lags / rolling stats**, linears
barely move. This is the core feature-engineering finding.

### 3.3 Failure analysis
`results/tables/failure_by_meter_engineered.csv`,
`failure_by_primary_use_engineered.csv`, plus the same for raw.
Steam meters are ~100x harder than electricity; RF wins every category.

### 3.4 Saved log-space predictions
`results/metrics/predictions_engineered/*_log_preds.npy` and
`results/metrics/predictions_raw/*_log_preds.npy`. Plus `y_true_raw.npy`.
`scripts/train_all.py` was patched to **resume from these** when present
(`--refit` to force retrain, `--only` / `--skip` for selective runs).

---

## 4. What is NOT done (the remaining work)

### 4.1 ARIMA + LSTM on engineered features (BLOCKING)
Two prior attempts:

- **First attempt** (`results/logs_remaining.txt`): ARIMA with `max_meters=60`
  ran 41 minutes, was silently SIGKILL'd (likely macOS OOM during stepwise
  search with `m=24`). `tee` masked the failure exit code — `set -eo pipefail`
  added in `scripts/run_finalize.sh`. LSTM never ran; later inspection
  exposed two bugs:
  - `best_val` was **never initialized** — would have crashed on epoch 0.
  - LSTM materialized `~26 GB` of training tensor (5.7M rows × 168 × 28 × 4B).
- **Fix already applied**:
  - `src/models/lstm.py` rewritten: meter subsampling via `max_meters`,
    striding via `stride`, scalar `best_val = float("inf")` init, per-epoch
    log line `LSTM ep=N val_mse=...`, fast `merge` filter instead of
    row-by-row tuple lookup.
  - `configs/default.yaml` LSTM block: `hidden_size: 96`, `num_layers: 1`,
    `batch_size: 512`, `epochs: 12`, `patience: 3`, `max_meters: 60`,
    `stride: 6`.
  - `configs/default.yaml` ARIMA block: `max_p: 2`, `max_q: 2`, `max_order: 4`,
    `stepwise: true`, `max_meters: 10` (last attempt was running fine but
    user stopped it at 42 min to context-switch — restart cleanly).
- **Second attempt** with `max_meters: 30` was killed manually after 42 min.
  Then restarted with `max_meters: 10` and killed almost immediately when
  the user asked to checkpoint. **No ARIMA / LSTM result row exists yet.**

### 4.2 Clustering experiment (BLOCKING)
`results/metrics/clustering.json` was produced but **the per-cluster RMSE is
broken** (77,953 vs global 5,731) because `scripts/run_clustering.py` line 87
used `np.clip(preds_log, a_min=None, a_max=20)` while everything else uses
`a_max=14.0`. `expm1(20) ≈ 485e6`, exploding the per-cluster aggregate.
**Fix already applied** (a_max=14.0). Needs to be re-run.

### 4.3 MLP investigation (NICE TO HAVE)
MLP RMSE=10,085 is much worse than RF (4,314) despite RMSLE only 0.90 vs 0.69.
Heavy-tail blow-ups on a few meters dominate RMSE. Optional: try a wider
clip range (currently `a_max=14.0` in `evaluate_predictions`), or a slightly
deeper / wider hidden stack. Not blocking — just be honest in the paper that
the MLP under-performs on heavy-tail steam meters.

### 4.4 Figures (BLOCKING for paper)
None of the publication figures exist yet. The `viz/` module is in place but
no `scripts/build_figures.py` driver. Required figures:

- EDA: hourly-mean profile by primary_use, per-meter-type distributions,
  weather correlations, missingness heatmap, target distribution log vs raw,
  per-site coverage timeline.
- Methods: feature-group diagram (can be drawn in TikZ, not required as PNG),
  K-means elbow plot, cluster centroids (24-hour profile per cluster).
- Results: model comparison bar chart (RMSE), per-meter-type RMSE heatmap,
  ablation Δ plot (raw vs engineered), per-cluster vs global LightGBM bar,
  residual histogram by hour-of-day, top-10 LightGBM feature importances.
- Failure: top-3 hardest primary_use × top-3 hardest meter heatmap.

Save **PDF for the paper**, **PNG for the README**. Use a single shared style
in `src/viz/style.py`.

### 4.5 Notebook execution (BLOCKING for repro)
The 8 notebooks in `notebooks/` exist as both `.py` (source) and `.ipynb`
(generated). They have **no executed outputs**. The grader expects to open
each notebook and see plots / tables inline. After all results are in, run
`jupyter nbconvert --to notebook --execute notebooks/0X_*.ipynb --inplace`
for each.

### 4.6 LaTeX paper (BLOCKING for grade)
**Not started yet — user explicitly deferred until code is fully done.**

NeurIPS-2024 style. 8 pages max. Sections:

1. Introduction & motivation (~0.5 pg)
2. Related work — ASHRAE GEPIII Kaggle solutions, BDG2 benchmark
   (Miller et al. 2020), prior building-energy ML surveys (~1.5 pg)
3. Data — ASHRAE GEPIII, 4-site subset, splits, cleaning decisions (~1 pg)
4. Methods — 10 models, 28 features grouped, clustering, baselines (~1.5 pg)
5. Results — main table, ablation, clustering, plots (~1.5 pg)
6. Analysis — failure modes, family wins/losses, gap to oracle (~1.5 pg)
7. Conclusion & limitations (~0.5 pg)

The paper must be written in **plain academic English with no AI tone**.
Use precise numbers from the saved CSVs / JSONs.

### 4.7 GitHub push (BLOCKING for grade)
- Currently only one commit: `86d5111 Initial scaffold: data, features, models, evaluation, viz`
- No remote configured (`git remote -v` returns nothing).
- Need to: add origin, stage everything except gitignored stuff, commit with
  a clean human message, push to `main`.
- After push, **open the GitHub contributors page** and verify only
  Akhilesh-Vangala appears. If anything else shows up, debug before
  submitting.

---

## 5. Known sharp edges

- **macOS multithreading + numpy / pmdarima / torch**: must export
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`
  before launching anything. `scripts/train_all.py` already does this.
  Do **not** call `torch.set_num_threads` from inside the LSTM model;
  it has caused segfaults.
- **`a_max` clip on log predictions**: the canonical value is **14.0**
  (`expm1(14) ≈ 1.2M kWh`, a sane physical upper bound). Anything higher
  blows up RMSE on tail meters.
- **Linear models**: `_LinearScaledMixin` filters zero-variance columns
  before scaling — **do not remove that**. It's why OLS/Ridge/Lasso are
  numerically stable on the small smoke set.
- **Random Forest**: `max_samples=0.5` in config is a deliberate speed knob.
- **ARIMA on stepwise auto search**: per-meter cost is unpredictable.
  `max_meters: 10` keeps wall time around 10–15 min; do not increase
  without budgeting.
- **Pandas merge in clustering**: `prep.val_full.merge(...)` returns a
  fresh range index; we depend on order preservation, which holds for
  one-to-one left merges but **not** in general. Verify any new joins.
- **`scripts/train_all.py` merges with prior `models_<set>.json`** if it
  exists and `--refit` is not passed. So `--only arima lstm` keeps the
  previous engineered table intact and only appends ARIMA + LSTM rows.

---

## 6. Acceptance criteria (definition of done)

- [ ] `results/tables/models_engineered.csv` includes rows for all 10 models
      (currently 8) plus 2 baselines.
- [ ] `results/metrics/clustering.json` has `per_cluster_overall.rmse` within
      a sensible factor (≤2x) of `global_lightgbm.rmse`.
- [ ] `results/figures/` populated with the figures listed in §4.4 (PDF + PNG).
- [ ] Every `.ipynb` in `notebooks/` has executed outputs and at least one
      figure rendered inline.
- [ ] `pytest -q` passes locally.
- [ ] `make all` runs end-to-end on a clean checkout (or, equivalently,
      `bash scripts/run_remaining.sh` followed by `bash scripts/run_finalize.sh`
      followed by figure / notebook scripts).
- [ ] `report/main.tex` compiles to `report/main.pdf`, exactly 8 pages,
      no overfull boxes.
- [ ] `git status` is clean. `git log` shows only Akhilesh-authored commits.
- [ ] `git push` to `Akhilesh-Vangala/Predicting-Building-Energy-Consumption`
      succeeds and the GitHub contributors graph shows only Akhilesh.

---

## 7. Recommended execution order

1. `cd "/Users/akhileshvangala/Desktop/Predicting Building Energy Consumption"`
2. `source .venv/bin/activate`
3. `bash scripts/run_finalize.sh` (ARIMA+LSTM+clustering, ~25 min)
4. Verify `results/tables/models_engineered.csv` now has 10 model rows.
5. Build figures: implement `scripts/build_figures.py` (use `src/viz/` helpers)
   and run it.
6. Execute notebooks: for each `notebooks/0X_*.ipynb`, run
   `jupyter nbconvert --to notebook --execute --inplace`.
7. `pytest -q` — must pass.
8. Write `report/main.tex` (NeurIPS style, 8 pages, plain English).
9. Compile `report/main.pdf`.
10. Stage, commit, set remote, push to `main`. Verify contributors graph.
