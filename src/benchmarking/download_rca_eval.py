import logging
import argparse
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure we can import from RCAEval
try:
    from RCAEval.utility import (
        download_re1_dataset,
        download_re2_dataset,
        download_re3_dataset,
    )
except ImportError:
    logging.error("RCAEval is not installed in the current environment.")
    logging.error("Please install it using: uv pip install -e './data/rca_eval_repo[default]'")
    sys.exit(1)

INCLUDE_RE1: bool = True
SKIP_RE2: bool = False
SKIP_RE3: bool = False

if __name__ == "__main__":
    # RCAEval utility functions automatically download to the "data" directory
    # relative to the current working directory.
    target_data_dir = Path("data")
    target_data_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Target data directory: {target_data_dir.resolve()}")

    if INCLUDE_RE1:
        logging.info("Downloading RE1 dataset (Metric-only data, 390MB)...")
        download_re1_dataset()
    else:
        logging.info("Skipping RE1 dataset (Set INCLUDE_RE1=True to download).")

    if not SKIP_RE2:
        logging.info("Downloading RE2 dataset (Multi-source data, ~4.2GB)...")
        download_re2_dataset()
    else:
        logging.info("Skipping RE2 dataset.")

    if not SKIP_RE3:
        logging.info("Downloading RE3 dataset (Multi-source code-level faults, 534MB)...")
        download_re3_dataset()
    else:
        logging.info("Skipping RE3 dataset.")

    logging.info("All selected RCAEval datasets have been downloaded and extracted.")
