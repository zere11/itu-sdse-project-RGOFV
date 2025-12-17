import pandas as pd
import numpy as np
from mlops_pipeline.modeling.save_best_model import DataPreprocessor

def test_preprocessor():
    # Simulate input data that caused issues
    df_train_raw = pd.DataFrame({
        "lead_id": [1, 2],
        "country": ["US", "UK"], # Should be dropped
        "onboarding": [True, False], # Should be normalized to 1, 0. Dummies -> onboarding_1, (0 dropped)
        "source": ["li", "fb"], # Dummies -> source_li, source_fb (one might be dropped depending on impl, usually alphabet first dropped if drop_first)
        # If source has 2 vals: fb, li. drop_first drops fb. Keeps source_li.
        "age": [30, 40]
    })
    
    # We need to determine "expected_columns" essentially by running the transform once "perfectly".
    # BUT DataPreprocessor is designed to TAKE expected_columns. 
    # Let's manually construct what X_train should look like OR use the class to generate it (if fit/transform separation was clean).
    # Since fit() in DataPreprocessor is trivial (just stores names), we can't use it to discover names.
    # We must assume an upstream process (prepare.py) generated X_train.
    
    # Let's verify the transform logic specifically.
    
    # 1. Run transform on TRAIN data without expected_columns enforcement to see what it produces.
    processor_dumb = DataPreprocessor(expected_columns=None)
    # fit doesn't do much without expected_columns
    processor_dumb.fit(df_train_raw) 
    
    # Transform
    X_train_transformed = processor_dumb.transform(df_train_raw)
    expected_cols = list(X_train_transformed.columns)
    
    print(f"Simulated X_train columns: {expected_cols}")
    
    # Now create the REAL processor for inference, passing expected_cols
    processor = DataPreprocessor(expected_columns=expected_cols)
    processor.fit(df_train_raw) # passed X doesn't matter much here since expected_cols is set
    
    # TEST: Inference with SINGLE row (onboarding=True)
    # This triggers the "drop_first" causing empty onboarding dummy issue.
    # We want to ensure restoration loop restores 'onboarding_1'.
    
    df_test_str = pd.DataFrame({
        "lead_id": [3],
        "country": ["DE"],
        "onboarding": ["True"], # String input, should be normalized to 1
        "source": ["organic"], 
        "age": [50]
    })
    
    trans_str = processor.transform(df_test_str)
    print(f"Transformed string input cols: {trans_str.columns.tolist()}")

    # Verifications
    if "country" not in trans_str.columns:
        print("PASS: country column was successfully dropped.")
    else:
         print("FAIL: country column persisted.")
    
    if "onboarding_True" in trans_str.columns:
        print("FAIL: onboarding_True column present (normalization failed).")
    else:
        print("PASS: onboarding_True column absent.")

    if "onboarding_1" in trans_str.columns:
        print("PASS: onboarding_1 column restored/present.")
    else:
        print("FAIL: onboarding_1 column missing (restoration failed).")


if __name__ == "__main__":
    try:
        test_preprocessor()
        print("Verification script finished.")
    except Exception as e:
        print(f"An error occurred: {e}")
