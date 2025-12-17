from pathlib import Path
import joblib
import shutil
import pandas as pd
from loguru import logger
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from mlops_pipeline.config import MODELS_DIR


def preprocess_for_inference(X, expected_columns=None):
    """
    Standalone preprocessing function for model inference.
    This function replicates all preprocessing steps from training.
    
    Args:
        X: Input DataFrame with raw features
        expected_columns: List of column names expected by the model (from training)
    
    Returns:
        DataFrame with processed features matching training data
    """
    data = X.copy()
    
    print(f"[preprocess_for_inference] Input shape: {data.shape}")
    print(f"[preprocess_for_inference] Input columns: {sorted(data.columns)[:10]}...")
    
    # Drop ID/leak columns
    cols_to_drop = ["lead_id", "customer_code", "date_part"]
    for col in cols_to_drop:
        if col in data.columns:
            data = data.drop(col, axis=1)
    
    # Drop columns that were removed during training
    cols_to_remove = ["is_active", "marketing_consent", "first_booking", 
                     "existing_customer", "last_seen", "domain", "country",
                     "visited_learn_more_before_booking", "visited_faq"]
    for col in cols_to_remove:
        if col in data.columns:
            data = data.drop(col, axis=1)
    
    # Create bin_source from source if it doesn't exist
    if "bin_source" not in data.columns and "source" in data.columns:
        data['bin_source'] = data['source']
        values_list = ['li', 'organic', 'signup', 'fb']
        data.loc[~data['source'].isin(values_list), 'bin_source'] = 'Others'
        mapping = {
            'li': 'socials', 
            'fb': 'socials', 
            'organic': 'group1', 
            'signup': 'group1'
        }
        data['bin_source'] = data['source'].map(mapping)
        data['bin_source'] = data['bin_source'].fillna('group1')
    
    # Normalize onboarding to boolean
    if "onboarding" in data.columns:
        vals = {
            1: True, 0: False,
            "1": True, "0": False,
            "True": True, "False": False, 
            "TRUE": True, "FALSE": False,
            True: True, False: False
        }
        data["onboarding"] = data["onboarding"].map(vals).fillna(False).astype(bool)
        # Explicitly set categories to ensure 'True' category exists
        data["onboarding"] = data["onboarding"].astype(
            pd.CategoricalDtype(categories=[False, True], ordered=False)
        )
    
    # One-hot encode categorical columns
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[[col for col in cat_cols if col in data.columns]].copy()
    other_vars = data.drop([col for col in cat_cols if col in data.columns], axis=1, errors='ignore')
    
    for col in cat_vars.columns:
        cat_vars[col] = cat_vars[col].astype("category")
        # Create dummy columns with drop_first=True
        df_dummies = pd.get_dummies(cat_vars[col], prefix=col, drop_first=True)
        cat_vars = pd.concat([cat_vars, df_dummies], axis=1)
        cat_vars = cat_vars.drop(col, axis=1)
    
    # Combine categorical and other variables
    data = pd.concat(
        [other_vars.reset_index(drop=True), cat_vars.reset_index(drop=True)], 
        axis=1
    )
    
    print(f"[preprocess_for_inference] After encoding shape: {data.shape}")
    print(f"[preprocess_for_inference] After encoding columns: {sorted(data.columns)[:10]}...")
    
    # Convert all columns to float
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    
    # 🔥 CRITICAL: Reindex to match training columns exactly
    if expected_columns is not None:
        print(f"[preprocess_for_inference] Reindexing to {len(expected_columns)} expected columns...")
        
        missing_cols = set(expected_columns) - set(data.columns)
        if missing_cols:
            print(f"[preprocess_for_inference] Missing columns (filling with 0): {sorted(missing_cols)}")
        
        extra_cols = set(data.columns) - set(expected_columns)
        if extra_cols:
            print(f"[preprocess_for_inference] Extra columns (dropping): {sorted(extra_cols)}")
        
        # Reindex guarantees exact column match with fill_value=0 for missing columns
        data = data.reindex(columns=expected_columns, fill_value=0.0)
        
        print(f"[preprocess_for_inference] Final shape: {data.shape}")
        print(f"[preprocess_for_inference] Columns match expected: {list(data.columns) == expected_columns}")
    
    return data


def save_best_model(
    xgboost_model,
    lr_model,
    xgboost_classification_report: dict,
    lr_classification_report: dict,
    xgboost_model_path: Path = None,
    lr_model_path: Path = None,
    output_path: Path = None,
    X_train: pd.DataFrame = None,
    printing: bool = False
) -> tuple[Path, str]:
    """
    Evaluate and save the best performing model (XGBoost vs Logistic Regression) as model.pkl.
    Compares models based on weighted average f1-score from classification reports.
    Wraps the model in a preprocessing pipeline to handle raw data during inference.
    
    Args:
        xgboost_model: Trained XGBoost model object
        lr_model: Trained Logistic Regression model object (can be a Pipeline)
        xgboost_classification_report: Classification report dict for XGBoost model
        lr_classification_report: Classification report dict for Logistic Regression model
        xgboost_model_path: Optional. Path to XGBoost model file (for copying if needed)
        lr_model_path: Optional. Path to Logistic Regression model file (for copying if needed)
        output_path: Optional. Path where to save the best model. Defaults to MODELS_DIR / "model.pkl"
        X_train: Training features DataFrame to get expected column names for preprocessing
        printing: Whether to print status messages
    
    Returns:
        tuple: (Path to saved model file, model type name)
    """
    if output_path is None:
        output_path = MODELS_DIR / "model.pkl"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract f1-scores from classification reports
    xgb_f1_score = xgboost_classification_report.get("weighted avg", {}).get("f1-score", 0.0)
    lr_f1_score = lr_classification_report.get("weighted avg", {}).get("f1-score", 0.0)
    
    if printing:
        logger.info(f"XGBoost weighted avg f1-score: {xgb_f1_score:.4f}")
        logger.info(f"Logistic Regression weighted avg f1-score: {lr_f1_score:.4f}")
    
    # Create preprocessing pipeline if X_train is provided
    if X_train is not None:
        logger.info(f"[save_best_model] X_train has {len(X_train.columns)} columns")
        logger.info(f"[save_best_model] X_train columns: {sorted(X_train.columns)[:10]}...")
        
        # Create FunctionTransformer with expected columns baked in via kw_args
        # This ensures the preprocessing function receives the expected columns during transform
        preprocessor = FunctionTransformer(
            func=preprocess_for_inference,
            kw_args={'expected_columns': list(X_train.columns)},
            validate=False  # Skip validation to allow DataFrames
        )
        
        # Compare and save the better model wrapped in a pipeline
        if xgb_f1_score >= lr_f1_score:
            # XGBoost is better or equal
            # XGBoost model is a simple estimator, wrap with preprocessor
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', xgboost_model)
            ])
            joblib.dump(pipeline, output_path)
            model_type = "XGBoost"
            if printing:
                logger.info(f"Best model: XGBoost (f1-score: {xgb_f1_score:.4f} >= {lr_f1_score:.4f})")
                logger.info(f"Saved best model (XGBoost) with preprocessing pipeline to {output_path}")
        else:
            # Logistic Regression is better
            # LR model from logreg.py is ALREADY a Pipeline with (imputer, logistic_regression)
            # We need to extract components and rebuild the pipeline
            if hasattr(lr_model, 'named_steps'):
                # lr_model is a Pipeline - extract the components
                logger.info("[save_best_model] LR model is a Pipeline, extracting components")
                final_estimator = lr_model.named_steps['logistic_regression']
                imputer = lr_model.named_steps['imputer']
                
                # Build new pipeline: Preprocessor -> Imputer -> LogisticRegression
                # This ensures raw data flows through all necessary transformations
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('imputer', imputer),
                    ('logistic_regression', final_estimator)
                ])
            else:
                # lr_model is a simple estimator (shouldn't happen with current code)
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', lr_model)
                ])
            
            joblib.dump(pipeline, output_path)
            model_type = "Logistic Regression"
            if printing:
                logger.info(f"Best model: Logistic Regression (f1-score: {lr_f1_score:.4f} > {xgb_f1_score:.4f})")
                logger.info(f"Saved best model (Logistic Regression) with preprocessing pipeline to {output_path}")
    else:
        # Fallback to original behavior if X_train not provided
        logger.warning("X_train not provided - saving model without preprocessing pipeline")
        if xgb_f1_score >= lr_f1_score:
            joblib.dump(xgboost_model, output_path)
            model_type = "XGBoost"
            if printing:
                logger.info(f"Best model: XGBoost (f1-score: {xgb_f1_score:.4f} >= {lr_f1_score:.4f})")
                logger.info(f"Saved best model (XGBoost) to {output_path}")
        else:
            if lr_model_path and lr_model_path.exists():
                shutil.copy(lr_model_path, output_path)
            else:
                joblib.dump(lr_model, output_path)
            model_type = "Logistic Regression"
            if printing:
                logger.info(f"Best model: Logistic Regression (f1-score: {lr_f1_score:.4f} > {xgb_f1_score:.4f})")
                logger.info(f"Saved best model (Logistic Regression) to {output_path}")
    
    return output_path, model_type