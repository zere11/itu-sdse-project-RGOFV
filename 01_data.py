## mathm: Helper functions.
import subprocess
import pandas as pd
import os
import datetime
import json
import warnings
import numpy as np



def describe_numeric_col(x):
    """
    Calculates various descriptive stats for a numeric column in a dataframe.

    Input:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """
    Imputes the mean/median for numeric columns or the mode for other types.


    Input:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    Output:
        x (pd.Series): Pandas series with added mean/median
    """
    if method not in ["mean", "median"]:
        raise ValueError("Method must be either mean or median.")

    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x



def dvc_pull():
    if "./artifacts/raw_data.csv" not in os.listdir("./artifacts"):
        print("DVC data not found locally. Pulling from remote storage...")
        try:
            result = subprocess.run(["dvc", "pull"], check=True, capture_output=True, text=True)
            print("DVC pull output:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error during DVC pull:")
            print(e.stderr)
    else:
        print("DVC data found locally. No need to pull.")

def data_extraction():
    data = pd.read_csv("./artifacts/raw_data.csv")
    max_date = "2024-01-31"
    min_date = "2024-01-01"


    if not max_date:
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()

    min_date = pd.to_datetime(min_date).date()

    # Time limit data
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    with open("./artifacts/date_limits.json", "w") as f:
        json.dump(date_limits, f)

def main():
    os.makedirs("artifacts",exist_ok=True)
    warnings.filterwarnings('ignore')
    pd.set_option('display.float_format',lambda x: "%.3f" % x)
    #print("Created artifacts directory")
    dvc_pull()
    data_extraction()
    print("Data extraction completed.")

if __name__ == "__main__":
    main()