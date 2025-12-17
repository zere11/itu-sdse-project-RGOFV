"""
Dataset loading utilities.

This module contains the raw data extraction logic exactly as implemented
in the original 01_data.py.
"""

import os
import pandas as pd


def data_extraction():
    """
    Look for raw_data.csv without using DVC.
    We run from /repo/notebooks, so the correct local path is:
      - ./artifacts/raw_data.csv
    Also allow:
      - ../artifacts/raw_data.csv  (repo root artifacts)
      - RAW_DATA_PATH env var (absolute override)
    """

    cwd = os.getcwd()  # should be /repo/notebooks
    candidates = [
        os.path.join(cwd, "artifacts", "raw_data.csv"),                # ./artifacts/raw_data.csv
        os.path.abspath(os.path.join(cwd, "..", "artifacts", "raw_data.csv")),  # ../artifacts/raw_data.csv
        os.environ.get("RAW_DATA_PATH", ""),                           # optional absolute path
    ]

    for path in candidates:
        if path and os.path.exists(path):
            print(f"✓ Loading data from: {path}")
            return pd.read_csv(path)
