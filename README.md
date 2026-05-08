# Predicting Building Energy Consumption

DS-GA 1003 — Machine Learning, NYU Spring 2026  
Akhilesh Vangala (sv3129) · Lucas Yao (ly2808) · Anvita Reddy (ari3289)

This is our final project for the Applied ML course at NYU CDS. We wanted to answer a pretty basic question that's surprisingly hard to answer from the existing literature: when you hold the features and data pipeline constant, how much does model choice actually matter for building energy forecasting?

Most prior work on the ASHRAE GEPIII dataset has teams varying their preprocessing, features, and models all at once, so you can't tell what's actually driving the results. We kept everything fixed — same 27 engineered features, same cleaning, same chronological split — and only changed the model.

We ended up comparing linear models (OLS, Ridge, Lasso, ElasticNet), tree ensembles (Decision Tree, Random Forest, LightGBM), a global LSTM, per-meter ARIMA, and a 3-layer MLP. We also ran a K-means clustering experiment that fits separate LightGBM models per consumption-profile cluster, and a per-use failure analysis to see where each family breaks down.

## Reproducing the results

Everything runs off a single config file. Once the data is in place, all the figures and tables in the paper come from running the scripts under `scripts/`.

### 1. Environment

Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` registers `src/` as the `bep` package so the notebooks and scripts can import it without sys.path hacks.

### 2. Data

Three files from the ASHRAE GEPIII Kaggle competition go into `data/raw/`:

```
data/raw/
  train.csv               # 20.2M hourly readings, 1,449 buildings
  building_metadata.csv   # site_id, primary_use, square_feet, year_built, floor_count
  weather_train.csv       # hourly weather per site
```

The competition is closed so you'll need to accept the late submission terms on Kaggle to unlock the download. After that you can pull files manually or run `python scripts/download_data.py` if you have `~/.kaggle/kaggle.json` set up.

We ran everything on a 4-site subset (sites 2, 4, 13, 14), which gives about 7.9 million rows after cleaning and covers all four meter types and all primary-use categories. It's manageable on a single workstation. You can switch to the full 20.2M row dataset with `data.use_full: true` in `configs/default.yaml` if you have the compute for it.

### 3. Pipeline

```bash
python -m scripts.build_features --config configs/default.yaml
python -m scripts.train_all      --config configs/default.yaml
python -m scripts.run_ablation   --config configs/default.yaml
python -m scripts.run_clustering --config configs/default.yaml
python -m scripts.run_failure_analysis --config configs/default.yaml
```

Or just `make all` to run everything in order. Results go to `results/`:

```
results/
  metrics/                # per-model JSON with RMSE, MAE, CV-RMSE, training time
  tables/                 # CSV tables used in the paper
  plots/                  # all figures from the paper
  models/                 # serialized model files (gitignored)
```

### 4. Notebooks

The notebooks follow the structure of the paper and are the easiest way to walk through the analysis:

| Notebook | What it covers |
|---|---|
| `01_eda.ipynb` | Distributions, missingness, temporal patterns, per-site/meter/use breakdowns |
| `02_feature_engineering.ipynb` | All 27 features and what each one captures |
| `03_linear_models.ipynb` | OLS, Ridge, Lasso, ElasticNet with time-series CV |
| `04_tree_models.ipynb` | Decision Tree, Random Forest, LightGBM with feature importance |
| `05_time_series.ipynb` | Per-meter ARIMA and the 168-hour global LSTM |
| `06_mlp.ipynb` | 3-layer MLP with dropout and early stopping |
| `07_clustering.ipynb` | K-means over 24h consumption profiles + per-cluster LightGBM |
| `08_analysis.ipynb` | 5-feature vs 27-feature ablation, per-type failure analysis |

## Models compared

| Family | Models |
|---|---|
| Linear | OLS, Ridge, Lasso, ElasticNet |
| Trees | Decision Tree, Random Forest, LightGBM |
| Time series | Per-meter ARIMA, global LSTM (168h lookback) |
| Neural | 3-layer MLP (256 / 128 / 64) |

All ten models train on the same 27 engineered features and the same time-based split (Jan–Sep 2016 train, Oct–Dec 2016 val). Any difference in validation RMSE is coming from the model, not from different preprocessing choices.

## Data

| Item | Value |
|---|---|
| Source | ASHRAE Great Energy Predictor III (Kaggle, 2019) |
| Full dataset | 20.2M hourly readings, 1,449 buildings, 16 sites, all of 2016 |
| Subset used | Sites 2, 4, 13, 14 — 482 buildings, ~7.9M rows after cleaning |
| Target | `meter_reading` in kWh, trained on `log1p(y)` |
| Split | Jan–Sep 2016 train, Oct–Dec 2016 val, strictly chronological |

Cleaning: zero-consumption streaks of 48+ consecutive hours are dropped as sensor outages, readings above the 99.9th percentile per (building, meter) are capped, and weather gaps are filled by linear interpolation within each site.

## Repo layout

```
.
├── configs/                 # YAML config for the whole pipeline
├── data/raw/                # raw CSVs go here (gitignored)
├── notebooks/               # 01–08, one notebook per analysis section
├── results/                 # metrics, tables, figures, models
├── scripts/                 # pipeline entry points
├── src/
│   ├── data/                # loading, cleaning, time-splitting
│   ├── features/            # 27 engineered features
│   ├── models/              # one module per model family
│   ├── evaluation/          # metrics and per-group breakdowns
│   ├── clustering/          # K-means over consumption profiles
│   ├── viz/                 # plotting helpers
│   ├── config.py
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
