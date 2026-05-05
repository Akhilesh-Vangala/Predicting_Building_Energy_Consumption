from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise ImportError(
            "Install the Kaggle CLI first: `pip install kaggle` and place "
            "your API key at ~/.kaggle/kaggle.json"
        ) from e

    api = KaggleApi()
    api.authenticate()

    competition = "ashrae-energy-prediction"
    target_files = ["train.csv", "building_metadata.csv", "weather_train.csv"]

    for fname in target_files:
        target = out / fname
        if target.exists():
            logger.info("%s already exists, skipping", fname)
            continue
        logger.info("Downloading %s", fname)
        api.competition_download_file(competition, fname, path=str(out))
        zipped = out / f"{fname}.zip"
        if zipped.exists():
            with zipfile.ZipFile(zipped, "r") as zf:
                zf.extractall(out)
            zipped.unlink()
        logger.info("Saved %s", target)


if __name__ == "__main__":
    main()
