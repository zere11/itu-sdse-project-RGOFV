from pathlib import Path

# Project root
# Goes up one level to parent directory from where config file is stored
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Directories
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
DATA_DIR = PROJ_ROOT / "data"

# Files
RAW_DATA_FILE = DATA_DIR / "raw_data.csv"
DATE_LIMITS_FILE = ARTIFACTS_DIR / "date_limits.json"
OUTLIER_SUMMARY_FILE = ARTIFACTS_DIR / "outlier_summary.csv"
CAT_MISSING_IMPUTE_FILE = ARTIFACTS_DIR / "cat_missing_impute.csv"
SCALER_FILE = ARTIFACTS_DIR / "scaler.pkl"
COLUMNS_DRIFT_FILE = ARTIFACTS_DIR / "columns_drift.json"
TRAINING_DATA_FILE = DATA_DIR / "training_data.csv"
TRAIN_GOLD_FILE = DATA_DIR / "train_data_gold.csv"
XGBOOST_MODEL_FILE = ARTIFACTS_DIR / "lead_model_xgboost.json"
LR_MODEL_FILE = ARTIFACTS_DIR / "lead_model_lr.pkl"
COLUMNS_LIST_FILE = ARTIFACTS_DIR / "columns_list.json"
MODEL_RESULTS_FILE = ARTIFACTS_DIR / "model_results.json"
X_TEST_FILE = DATA_DIR / "X_test.csv"
Y_TEST_FILE = DATA_DIR / "y_test.csv"
DATE_FILTERED_DATA_FILE = DATA_DIR / "date_filtered_data.csv"

# MLFlow registry
MLRUNS_DIR = PROJ_ROOT / "mlruns"
MLRUNS_TRASH_DIR = MLRUNS_DIR / ".trash"