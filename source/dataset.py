import os
import pandas as pd
from pathlib import Path

def data_extraction():
    """
    Load raw_data.csv from artifacts directory.
    When running from /repo/source, artifacts is at ../artifacts
    """
    candidates = [
        os.path.join("../artifacts", "raw_data.csv"),                  # ../artifacts/raw_data.csv
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "raw_data.csv")),
        os.environ.get("RAW_DATA_PATH", ""),                           # optional absolute path
    ]

    for path in candidates:
        if path and os.path.exists(path):
            print(f"Loading data from: {path}")
            return pd.read_csv(path)
    
    raise FileNotFoundError(f"Could not find raw_data.csv in any of: {candidates}")

# Project root
# Goes up one level to parent directory from where config file is stored
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Directories
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
DATA_DIR = PROJ_ROOT / "data"