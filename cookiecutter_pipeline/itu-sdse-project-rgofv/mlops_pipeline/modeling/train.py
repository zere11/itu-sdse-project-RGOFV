from pathlib import Path
import pandas as pd

from loguru import logger
from tqdm import tqdm
import typer

from mlops_pipeline.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from mlops_pipeline.modeling.xgboost import train_xgboost_model

app = typer.Typer()


# We need some reader functions: 
def _load_X_y(X_path: Path, y_path: Path):
    """
    Load features and single-column labels from CSV.
    Assumes index is in column 0; adjust index_col as needed.
    """
    X = pd.read_csv(X_path, index_col=0)
    y_df = pd.read_csv(y_path, index_col=0)

    if y_df.shape[1] != 1:
        raise ValueError(f"{y_path} must have exactly 1 column, got {y_df.shape[1]}")

    y = y_df.iloc[:, 0]
    # Ensure binary labels are ints 0/1 if applicable
    if set(y.dropna().unique()) <= {0, 1}:
        y = y.astype(int)

    return X, y




@app.command()
def main(
    X_train_path: Path = INTERIM_DATA_DIR / "X_train.csv",
    X_test_path: Path = INTERIM_DATA_DIR / "X_test.csv",
    y_train_path: Path = INTERIM_DATA_DIR / "y_train.csv",
    y_test_path: Path = INTERIM_DATA_DIR / "y_test.csv",

    xgboost_pkl_path: Path = MODELS_DIR / "lead_model_xgboost.pkl",
    xgboost_json_path: Path = MODELS_DIR / "lead_model_xgboost.json",

    random_state: int = 42,
    printing_bool: bool = False,

):
    '''
    Train XGBoost RF using the pre-split csv's and saving a pkl and json of the best model.
    '''
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Loading the split csv's...")
    X_train, y_train = _load_X_y(X_train_path, y_train_path)
    X_test,  y_test  = _load_X_y(X_test_path,  y_test_path)


    logger.info("Training XGBRFClassifier (randomized search)...")
    model_path, model_results, best_model = train_xgboost_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        xgboost_model_path=xgboost_pkl_path,
        xgboost_json_path=xgboost_json_path,
        random_state=random_state,
        printing=printing_bool,
    )


    logger.success(f"Saved scikit-learn wrapper to: {xgboost_model_path}")
    logger.success(f"Saved native XGBoost Booster JSON to: {xgboost_json_path}")






    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
