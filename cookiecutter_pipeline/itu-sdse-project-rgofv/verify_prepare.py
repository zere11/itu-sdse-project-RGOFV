import pandas as pd
import numpy as np

def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

def test_onboarding_dummies():
    # Simulate input
    df = pd.DataFrame({
        "onboarding": [1, 0, 1]
    })
    
    # Logic from prepare.py
    col = "onboarding"
    df[col] = df[col].astype(bool)
    df[col] = df[col].astype("category")
    
    print(f"Values after bool cast: {df[col].tolist()}")
    
    df_out = create_dummy_cols(df, col)
    print("Columns:", df_out.columns.tolist())
    
    if "onboarding_True" in df_out.columns:
        print("SUCCESS: onboarding_True generated.")
    else:
        print("FAIL: onboarding_True NOT generated.")
        
if __name__ == "__main__":
    test_onboarding_dummies()
