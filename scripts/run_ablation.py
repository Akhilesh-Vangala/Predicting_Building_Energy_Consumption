from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from src.config import load_config
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    for fs in ["raw", "engineered"]:
        logger.info("=== running ablation with feature_set=%s ===", fs)
        cmd = [sys.executable, "-m", "scripts.train_all",
               "--config", args.config, "--feature-set", fs,
               "--skip", "lstm", "arima"]
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
