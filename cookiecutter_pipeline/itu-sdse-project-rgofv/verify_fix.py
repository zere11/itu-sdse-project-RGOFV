import pandas as pd
import numpy as np
from mlops_pipeline.modeling.save_best_model import DataPreprocessor

def test_preprocessor():
    # Simulate input data that caused issues
    df_train_raw = pd.DataFrame({
        "lead_id": [1, 2],
        "country": ["US", "UK"], # Should be dropped
        "onboarding": [True, False], # Should be normalized
        "source": ["li", "fb"],
        "age": [30, 40]
    })
    
    # Expected behavior after my fix:
    # 1. country dropped
    # 2. onboarding -> 1/0 -> get_dummies -> onboarding_1, (onboarding_0 dropped if drop_first=True)
    # Wait, get_dummies(drop_first=True) on [1, 0] produces 'col_1'. 
    # If it was True/False, get_dummies might produce 'col_True'.
    
    processor = DataPreprocessor()
    processor.fit(df_train_raw)
    
    transformed = processor.transform(df_train_raw)
    
    print(f"Transformed columns: {transformed.columns.tolist()}")
    
    if "country" not in transformed.columns:
        print("PASS: country column was successfully dropped.")
    else:
         print("FAIL: country column persisted.")

    cols = transformed.columns.tolist()
    has_onboarding_1 = any("onboarding_1" in c for c in cols)
    has_onboarding_True = any("onboarding_True" in c for c in cols)
    
    print(f"Has onboarding_1: {has_onboarding_1}")
    print(f"Has onboarding_True: {has_onboarding_True}")
    
    if has_onboarding_True:
        print("FAIL: onboarding was treated as boolean/string True")
    if has_onboarding_1:
         print("SUCCESS: onboarding was treated as int 1")

    # verify consistency
    df_test_str = pd.DataFrame({
        "lead_id": [3],
        "country": ["DE"],
        "onboarding": ["True"],
        "source": ["organic"],
        "age": [50]
    })
    trans_str = processor.transform(df_test_str)
    print(f"Transformed string input cols: {trans_str.columns.tolist()}")
    
    if "onboarding_1" in trans_str.columns:
        print("SUCCESS: string 'True' converted to 1 and encoded correctly")
    else:
        print("FAIL: string 'True' not handled correctly")

if __name__ == "__main__":
    try:
        test_preprocessor()
        print("Verification script finished.")
    except Exception as e:
        print(f"An error occurred: {e}")
