# Raw data

Three files from the ASHRAE Great Energy Predictor III Kaggle competition go in this folder:

| File | Size | Notes |
|------|------|-------|
| `train.csv` | ~647 MB | 20.2M hourly meter readings |
| `building_metadata.csv` | ~44 KB | site_id, primary_use, square_feet, year_built, floor_count |
| `weather_train.csv` | ~14 MB | hourly weather per site (16 sites) |

The competition is closed; access requires accepting the rules through Kaggle's *Late
Submission* button on the [competition data page](https://www.kaggle.com/competitions/ashrae-energy-prediction/data),
which is a one-time step.

After that:

- **CLI:** put `~/.kaggle/kaggle.json` in place and run `python scripts/download_data.py`
- **Manual:** download the three CSVs from the Kaggle data tab and drop them in this folder.

The other Kaggle files (`test.csv`, `weather_test.csv`, `sample_submission.csv`) are not
used by this project.
