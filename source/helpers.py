import pandas as pd
import numpy as np

def describe_numeric_col(x):
    return pd.Series([x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
                     index=["Count", "Missing", "Mean", "Min", "Max"])

def impute_missing_values(x, method="mean"):
    if method not in ["mean", "median"]:
        raise ValueError("Method must be either mean or median.")
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x

def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df
