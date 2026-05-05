# Predicting Building Energy Consumption

DS-GA 1003 — Machine Learning, NYU Spring 2026 (Applied ML track)
Akhilesh Vangala (sv3129) · Lucas Yao (ly2808) · Anvita Reddy (ari3289)

We forecast hourly meter-level energy consumption for 1,449 buildings in the ASHRAE
Great Energy Predictor III dataset and compare ten methods drawn from four model
families — linear, tree ensembles, classical time series, and neural networks — on a
shared feature set and time-based split. We then run a clustering experiment that groups
buildings by their hour-of-day consumption profile, and a per-building-type / per-meter-type
failure analysis to characterise where each family breaks down.

The full proposal is in `report/proposal.pdf`. The final paper is in `report/report.pdf`.

## Reproducing the results

The whole pipeline is meant to be reproducible from a single environment and a single
config file. Once the data is in place, every figure and table in the report comes from
`make all` or the equivalent script under `scripts/`.

### 1. Environment

Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` registers the `src/` package as `bep` so the notebooks and scripts can
import it without `sys.path` hacks.

### 2. Data

Three files from the ASHRAE GEPIII Kaggle competition go into `data/raw/`:

```
data/raw/
  train.csv               # 20.2M hourly meter readings, 1,449 buildings
  building_metadata.csv   # site_id, primary_use, square_feet, year_built, floor_count
  weather_train.csv       # hourly weather per site
```

The competition is closed, so Kaggle requires a one-time *Late Submission* acknowledgement
to unlock the download. Once that is done you can either pull the files manually or run
`python scripts/download_data.py` (needs `~/.kaggle/kaggle.json`).

The default config uses a 4-site subset (sites 2, 4, 13, 14) — about 3 million rows after
cleaning — which keeps every model trainable on a single workstation. Switching
`data.use_full: true` in `configs/default.yaml` runs on the entire 20.2M-row dataset.

### 3. Pipeline

```bash
python -m scripts.build_features --config configs/default.yaml
python -m scripts.train_all      --config configs/default.yaml
python -m scripts.run_ablation   --config configs/default.yaml
python -m scripts.run_clustering --config configs/default.yaml
python -m scripts.run_failure_analysis --config configs/default.yaml
```

`make all` runs all of the above in order. Outputs land in `results/`:

```
results/
  metrics/                # per-model JSON with RMSE, CV-RMSE, MAE, train time
  tables/                 # publication-ready CSV/markdown tables
  plots/                  # every figure used in the report
  models/                 # serialised models (gitignored)
```

### 4. Notebooks

The `notebooks/` folder mirrors the report and is the easiest way to follow the analysis
end to end:

| Notebook | What it covers |
|---|---|
| `01_eda.ipynb` | Distributions, missingness, temporal patterns, per-site / per-meter / per-use breakdowns |
| `02_feature_engineering.ipynb` | Walks through all 28 features and shows what each one captures |
| `03_linear_models.ipynb` | OLS, Ridge, Lasso, ElasticNet with time-series CV |
| `04_tree_models.ipynb` | Decision Tree, Random Forest, LightGBM with feature importance |
| `05_time_series.ipynb` | Per-meter ARIMA and 168-hour LSTM |
| `06_mlp.ipynb` | 3-layer MLP with dropout and early stopping |
| `07_clustering.ipynb` | K-means over hour-of-day profiles + per-cluster LightGBM |
| `08_analysis.ipynb` | 5-feature vs 28-feature ablation, per-type failure analysis, gap analysis |

## Methods compared

| Family | Models |
|---|---|
| Linear | OLS, Ridge, Lasso, ElasticNet |
| Trees | Decision Tree, Random Forest, LightGBM |
| Time series | Per-meter ARIMA, LSTM (168h lookback) |
| Neural | 3-layer MLP (256 / 128 / 64) |

All ten models train on the same 28 engineered features and the same time-based split
(Jan–Sep 2016 train, Oct–Dec 2016 validation). LightGBM additionally serves as the
backbone for the per-cluster clustering experiment.

## Data

| Item | Value |
|---|---|
| Source | ASHRAE Great Energy Predictor III (Kaggle, 2019) |
| Coverage | All of 2016, 16 sites, 1,449 buildings, 20.2M hourly readings |
| Subset used | 4 sites (2, 4, 13, 14), ≈3M rows after cleaning |
| Target | `meter_reading` in kWh, trained on `log1p(y)` |
| Split | Train Jan–Sep 2016, validate Oct–Dec 2016, no shuffling |

Cleaning: site 0 electricity readings are converted from kBTU to kWh, zero-streaks of
≥48 consecutive hours are flagged as outages and removed, readings above the 99.9th
per-meter quantile are capped, and weather gaps are forward filled per site.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── setup.py
├── Makefile
├── configs/                 # YAML configs for the whole pipeline
├── data/
│   └── raw/                 # CSVs go here (gitignored)
├── notebooks/               # 01–08, mirror the report sections
├── report/                  # proposal PDF, final paper LaTeX + PDF
├── results/                 # generated metrics, tables, plots, models
├── scripts/                 # CLI entry points for each pipeline stage
├── src/
│   ├── data/                # load, clean, time-split
│   ├── features/            # 28 engineered features
│   ├── models/              # one module per family + a shared base
│   ├── evaluation/          # metrics + per-group breakdowns
│   ├── clustering/          # K-means consumption profiles
│   ├── viz/                 # plotting helpers
│   ├── config.py            # config dataclasses
│   └── utils.py
└── tests/                   # unit tests (leakage, split, features)
```

## References

- Miller, C. et al. (2020). The ASHRAE Great Energy Predictor III competition: overview and results. *Sci. Technol. Built Environ.* 26(10), 1427–1447.
- Miller, C., Hao, L., Fu, C. (2022). Gradient boosting machines and careful pre-processing work best: ASHRAE GEPIII lessons. *ASHRAE Trans.* 128, 405–413.
- Miller, C., Picchetti, B., Fu, C., Pantelic, J. (2022). Limitations of machine learning for building energy prediction: ASHRAE GEPIII error analysis. *Sci. Technol. Built Environ.* 28(8), 1036–1052.
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS.*
- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS.*
- Runge, J., Zmeureanu, R. (2019). Forecasting energy use in buildings using artificial neural networks: a review. *Energies* 12(18), 3355.
