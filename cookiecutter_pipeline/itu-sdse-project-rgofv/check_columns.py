from pathlib import Path
import pandas as pd
import sys
import os

# Add current directory to path so we can import mlops_pipeline
sys.path.append(os.getcwd())

try:
    from mlops_pipeline.config import INTERIM_DATA_DIR
except ImportError:
    # If import fails, try to deduce path manually
    INTERIM_DATA_DIR = Path("data/interim")

try:
    path = INTERIM_DATA_DIR / "X_train.csv"
    print(f"Reading from: {path.absolute()}")
    X_train = pd.read_csv(path)
    print("Columns in X_train.csv:")
    for col in X_train.columns:
        print(f"- {col}")
except Exception as e:
    print(f"Error reading X_train.csv: {e}")
