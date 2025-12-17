from pathlib import Path
import joblib
import shutil
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from mlops_pipeline.config import MODELS_DIR


def create_dummy_cols(df, col):
    '''
    Create one-hot encoding columns in the data.
    '''
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df


class RobustDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to preprocess data for model inference.
    Handles the same preprocessing steps as the training pipeline.
    Renamed to RobustDataPreprocessor to ensure fresh code usage.
    """
    def __init__(self, expected_columns=None):
        self.expected_columns = expected_columns
        # Use simple attribute name to ensure persistence across sklearn versions
        self.final_columns = None
        
    def fit(self, X, y=None):
        # Store expected feature names from training
        if self.expected_columns is not None:
            self.final_columns = list(self.expected_columns)
        else:
            self.final_columns = list(X.columns)
        
        logger.info(f"[RobustDataPreprocessor] Fitted with {len(self.final_columns)} columns")
        logger.info(f"[RobustDataPreprocessor] Expected columns: {sorted(self.final_columns)[:10]}...")
        return self
    
    def transform(self, X):
        # Make a copy to avoid modifying original
        data = X.copy()
        
        logger.info(f"[RobustDataPreprocessor] Transform input shape: {data.shape}")
        logger.info(f"[RobustDataPreprocessor] Transform input columns: {sorted(data.columns)[:10]}...")
        
        # Drop columns that should not be in the final features
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
        
        # Create bin_source from source if it doesn't exist (as done in data_preparation)
        if "bin_source" not in data.columns and "source" in data.columns:
            data['bin_source'] = data['source']
            values_list = ['li', 'organic', 'signup', 'fb']
            data.loc[~data['source'].isin(values_list), 'bin_source'] = 'Others'
            mapping = {'li': 'socials', 
                      'fb': 'socials', 
                      'organic': 'group1', 
                      'signup': 'group1'}
            data['bin_source'] = data['source'].map(mapping)
            # Fill any NaN values with 'group1' as default
            data['bin_source'] = data['bin_source'].fillna('group1')
        
        # Normalize onboarding
        # We align with prepare.py which forces boolean -> onboarding_True
        if "onboarding" in data.columns:
             # Handle various input types: boolean, string, int -> map to BOOL
             vals = {
                 1: True, 0: False,
                 "1": True, "0": False,
                 "True": True, "False": False, "TRUE": True, "FALSE": False,
                 True: True, False: False
             }
             data["onboarding"] = data["onboarding"].map(vals).fillna(False).astype(bool)
             
             # Explicitly set categories to ensure 'True' category exists (matches prepare.py)
             data["onboarding"] = data["onboarding"].astype(pd.CategoricalDtype(categories=[False, True], ordered=False))

        # Handle categorical columns - one-hot encode
        cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
        cat_vars = data[[col for col in cat_cols if col in data.columns]].copy()
        other_vars = data.drop([col for col in cat_cols if col in data.columns], axis=1, errors='ignore')
        
        for col in cat_vars.columns:
            cat_vars[col] = cat_vars[col].astype("category")
            cat_vars = create_dummy_cols(cat_vars, col)
        
        # Combine
        data = pd.concat([other_vars.reset_index(drop=True), cat_vars.reset_index(drop=True)], axis=1)
        
        logger.info(f"[RobustDataPreprocessor] After one-hot encoding, columns: {sorted(data.columns)[:10]}...")
        logger.info(f"[RobustDataPreprocessor] Shape before reindex: {data.shape}")
        
        # Convert to float
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        
        # 🔥 CRITICAL FIX: Reindex to match training columns EXACTLY
        if self.final_columns is not None:
            logger.info(f"[RobustDataPreprocessor] Reindexing to {len(self.final_columns)} expected columns...")
            
            # Check which columns are missing
            missing_cols = set(self.final_columns) - set(data.columns)
            if missing_cols:
                logger.warning(f"[RobustDataPreprocessor] Missing columns (will be filled with 0): {sorted(missing_cols)}")
            
            # Check which columns are extra
            extra_cols = set(data.columns) - set(self.final_columns)
            if extra_cols:
                logger.warning(f"[RobustDataPreprocessor] Extra columns (will be dropped): {sorted(extra_cols)}")
            
            # Reindex guarantees exact column match with training
            data = data.reindex(columns=self.final_columns, fill_value=0.0)
            
            logger.info(f"[RobustDataPreprocessor] After reindex shape: {data.shape}")
            logger.info(f"[RobustDataPreprocessor] Final columns match training: {list(data.columns) == self.final_columns}")
        
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
        lr_model: Trained Logistic Regression model object
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
        # 🔥 CRITICAL: Pass the exact training column names
        logger.info(f"[save_best_model] X_train has {len(X_train.columns)} columns")
        logger.info(f"[save_best_model] X_train columns: {sorted(X_train.columns)[:10]}...")
        
        preprocessor = RobustDataPreprocessor(expected_columns=list(X_train.columns))
        preprocessor.fit(X_train)
        
        # Compare and save the better model wrapped in a pipeline
        if xgb_f1_score >= lr_f1_score:
            # XGBoost is better or equal
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
        if xgb_f1_score >= lr_f1_score:
            # XGBoost is better or equal
            joblib.dump(xgboost_model, output_path)
            model_type = "XGBoost"
            if printing:
                logger.info(f"Best model: XGBoost (f1-score: {xgb_f1_score:.4f} >= {lr_f1_score:.4f})")
                logger.info(f"Saved best model (XGBoost) to {output_path}")
        else:
            # Logistic Regression is better
            if lr_model_path and lr_model_path.exists():
                # Copy the existing pickle file if path is provided
                shutil.copy(lr_model_path, output_path)
            else:
                # Otherwise dump the model object
                joblib.dump(lr_model, output_path)
            model_type = "Logistic Regression"
            if printing:
                logger.info(f"Best model: Logistic Regression (f1-score: {lr_f1_score:.4f} > {xgb_f1_score:.4f})")
                logger.info(f"Saved best model (Logistic Regression) to {output_path}")
    
    return output_path, model_type