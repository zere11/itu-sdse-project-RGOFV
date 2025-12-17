import pandas as pd
import numpy as np

def test_build_loss():
    # Simulate CSV content where onboarding is boolean-like
    # When written as object "True"/"False" and read back, pandas often infers bool.
    from io import StringIO
    csv_data = """age,onboarding,source
25,True,organic
30,False,li
"""
    df = pd.read_csv(StringIO(csv_data))
    print("Types after read_csv:")
    print(df.dtypes)
    
    cont = df.select_dtypes(include=[np.number]).copy()
    cat = df.select_dtypes(include=["object"]).copy()
    
    df_feat = pd.concat([cat, cont], axis=1)
    
    print("\nColumns after split/concat:")
    print(df_feat.columns.tolist())
    
    if "onboarding" not in df_feat.columns:
        print("\nCONFIRMED: onboarding was dropped because it is bool!")
    else:
        print("\nFAILED TO REPRODUCE: onboarding is present.")

if __name__ == "__main__":
    test_build_loss()
