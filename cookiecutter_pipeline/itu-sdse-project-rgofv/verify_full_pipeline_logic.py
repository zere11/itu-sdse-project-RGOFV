import pandas as pd
import numpy as np
import sys
from mlops_pipeline.modeling.save_best_model import DataPreprocessor

def test_full_pipeline_logic():
    print("--- Starting Pipeline Logic Verification ---")

    # 1. Simulate 'prepared' X_train (as it comes from prepare.py)
    # Note: prepare.py creates dummies. So 'onboarding' is ALREADY dummies in X_train.
    # In legacy/prepare.py, create_dummy_cols(drop_first=True) is used.
    # If onboarding has vals {True, False} or {1, 0}:
    #  - If 1/0: defaults to int. get_dummies -> onboarding_1 (if 0 is ref)
    #  - If True/False: defaults to bool. get_dummies -> onboarding_True (if False is ref)
    
    # OUR FIX relies on prepare.py producing 'onboarding_1'.
    # Because prepare.py converts everything to float64/int at the end usually.
    # Let's assume the Training Data (X_train) looks like this:
    X_train_cols = [
        "age", 
        "onboarding_True",  # The result of dummy encoding with bool
        "source_li", "source_organic", "source_signup", # bin_source dummies
        # country is NOT here because we dropped it in train.py!
        # wait, train.py loads X_train then drops it.
        # But DataPreprocessor "expected_columns" comes from the X_train PASSED to it.
        # So if we drop in train.py before creating DataPreprocessor, 
        # expected_columns will NOT contain 'country'. CORRECT.
    ]
    
    # So the expected columns passed to DataPreprocessor will be:
    expected_cols = X_train_cols
    print(f"Expected Training Columns: {expected_cols}")

    # 2. Initialize Preprocessor with these columns
    processor = DataPreprocessor(expected_columns=expected_cols)
    # Fit is trivial (just sets feature_names_)
    processor.fit(None) # Argument ignored if expected_columns provided

    # 3. Simulate Inference Input (Raw Data)
    # The API/Inference receives RAW data, where 'onboarding' might be boolean True/False.
    # And 'country' is still present.
    inference_row = pd.DataFrame({
        "lead_id": [123], # Should be dropped
        "country": ["US"], # Should be dropped
        "onboarding": [True], # BOOLEAN input - This caused the crash!
        "source": ["organic"], 
        "age": [35],
        # potential missing columns
    })

    print("\nInference Input (Raw):")
    print(inference_row)

    # 4. Transform
    try:
        X_out = processor.transform(inference_row)
        print("\nTransform Output Columns:")
        print(X_out.columns.tolist())
        print("\nTransform Output Values:")
        print(X_out.values)
        
        # 5. Assertions
        cols = X_out.columns.tolist()
        
        # Check 1: Dropped columns
        if "country" in cols:
            print("FAIL: 'country' should have been dropped.")
            sys.exit(1)
        if "lead_id" in cols:
            print("FAIL: 'lead_id' should have been dropped.")
            sys.exit(1)
            
        # Check 2: Onboarding Normalization
        # We expect 'onboarding_True' to be present and 1.0 (since input was True)
        # Because we switched to Boolean normalization in save_best_model.py
        if "onboarding_True" not in cols:
            print("FAIL: 'onboarding_True' missing. Boolean Normalization failed.")
            if "onboarding_1" in cols:
                print("FAIL: Generated 'onboarding_1' instead! Int normalization logic persisted?")
            sys.exit(1)
            
        val_onboarding = X_out.iloc[0]["onboarding_True"]
        if val_onboarding != 1.0:
             print(f"FAIL: onboarding_True value is {val_onboarding}, expected 1.0")
             sys.exit(1)

        # Check 3: Missing columns filled
        if "source_li" not in cols:
             print("FAIL: 'source_li' (missing from input) should have been added by reindex.")
             sys.exit(1)
        
        val_li = X_out.iloc[0]["source_li"]
        if val_li != 0.0:
            print(f"FAIL: source_li should be 0.0, got {val_li}")
            sys.exit(1)

        print("\nSUCCESS: Pipeline logic verified!")
        
    except Exception as e:
        print(f"\nCRITICAL FAIL: Exception during transform: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_full_pipeline_logic()
