from __future__ import annotations
import pandas as pd
from pathlib import Path
import datetime
import json
import numpy as np
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
import joblib
from loguru import logger
import argparse


# Internal imports
from mlops_pipeline.config import DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PRINTING_STATE, MODELS_DIR


# Two helper functions for the data preparation.
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



def make_dataset(raw_csv_path: Path, out_training_csv_path: Path, min_date: str, max_date: str | None, printing: bool) -> None:
    '''
    Function for importing  the raw dataset and applying filters to have the training data ready for feature building.
    
    Input:
        raw_csv_path          - The pathlib Path to the raw csv
        out_training_csv_path - The pathlib Path to where we save the cleaned dataset, used for feature building
        min_date              - Minimum str date to include from the raw dataset. Can not be empty.
        max_date              - Maximum str date to include from the raw dataset. If left empty, will set to max date in dataset.
    
    Output:
        No direct output, but files created for features/build_features.py
    '''
    logger.info(f"Reading raw data: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    # date filtering
    if "date_part" in df.columns:
        df["date_part"] = pd.to_datetime(df["date_part"]).dt.date
        min_d = pd.to_datetime(min_date).date()
        max_d = pd.to_datetime(max_date).date() if max_date else df["date_part"].max()
        df = df[(df["date_part"] >= min_d) & (df["date_part"] <= max_d)]
        limits = {"min_date": str(df["date_part"].min()), "max_date": str(df["date_part"].max())}
        # We dump the json file of the date limits in the interim data folder. (will create folder if it is absent)
        (INTERIM_DATA_DIR / "metadata").mkdir(parents=True, exist_ok=True)
        with open(INTERIM_DATA_DIR / "metadata" / "date_limits.json", "w") as f:
            json.dump(limits, f, indent=2)

    # Fill up empty fields with NaN
    for col in ["lead_indicator", "lead_id", "customer_code"]:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan)
    # We remove rows where lead_indicator or lead_id is NaN
    df = df.dropna(axis=0, subset=[c for c in ["lead_indicator", "lead_id"] if c in df.columns])

    # We want to make sure the data in these columns is understood as pandas objects (so not continuous or time-based)
    obj_cols = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in obj_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # Now we can take all the numeric columns, and then capping the outlier values to within mean +/- 2 standard deviations. 
    cont = df.select_dtypes(include=[np.number]).copy()
    cont = cont.apply(lambda x: x.clip(lower=x.mean() - 2*x.std(), upper=x.mean() + 2*x.std()))  # x here is a pandas Series for each column
    # And then for the missing values, we add the mean (would be mode if categorical)
    cont = cont.apply(impute_missing_values)

    # Now we take all the object columns as defined above.
    cat = df.select_dtypes(include=["object"]).copy()
    if "customer_code" in cat.columns:
        cat.loc[cat["customer_code"].isna(), "customer_code"] = "None"  # For customer_code specifically, we set missing values to None
    # For NaN in object columns, we assign the mode (most frequent category for that column)
    cat = cat.apply(impute_missing_values)

    # Now we recombine the whole thing.
    df_clean = pd.concat([cat.reset_index(drop=True), cont.reset_index(drop=True)], axis=1)

    # This we can output as the training data set for feature building.
    out_training_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_training_csv_path, index=False)

    # We save the columns as metadata, to test for column drift in case we get new data in the future.
    with open(INTERIM_DATA_DIR / "metadata" / "columns_drift.json", "w") as f:
        json.dump(list(df_clean.columns), f, indent=2)

    # As always, we can print.
    if printing:
        logger.info(f"Interim saved: {out_training_csv_path} (rows={len(df_clean)}, cols={df_clean.shape[1]})")

def main():
    # We use the parser here so that code can be configurable from the terminal. Defaults set as we have them in the original script.
    parser = argparse.ArgumentParser(description="Prepares and makes the dataset for feature building from the raw_data.csv")
    parser.add_argument("--raw", type=Path, default=RAW_DATA_DIR / "raw_data.csv", help="Sets the path to the raw_data.csv. Default is in data/raw")
    parser.add_argument("--out", type=Path, default=INTERIM_DATA_DIR / "training_data.csv", help="Sets the path for the output of the data cleaning. Default is in data/interim")
    parser.add_argument("--min-date", type=str, default="2024-01-01", help="Sets the minimum date to be handled (Using format YYYY-MM-DD). Must be defined, default is 2024-01-01")
    parser.add_argument("--max-date", type=str, default="2024-01-31", help="Sets the maximum date to be handled (Using format YYYY-MM-DD). Defaults to 2024-01-31, but if explicitly undefined, max will be set to the latest date in raw_data.csv")
    parser.add_argument("--printing", action="store_true", default=PRINTING_STATE, help="Sets the printing state for logger, to see which parts are running. Default set by flag in mlops_pipeline/config.py as False. If called by --printing, will be set to true.")
    args = parser.parse_args()
    make_dataset(args.raw, args.out, args.min_date, args.max_date, args.printing)

if __name__ == "__main__":
    main()




# OLD FUNCTIONS - weave into the above
def data_extraction():
    '''
    Quick pointer to the raw data in the top level data folder. DVC functionality has been deprecated, but can be found in the original 01_data.py script.

    Output:
        data - A pandas dataframe of the raw dataset.
    '''
    
    # Removed earlier code for dvc functionality, as it requires a proper storage environment (Azure, etc)
    # The file is located directly in the raw folder under data.
    
    raw_path = RAW_DATA_DIR / "raw_data.csv"
    data = pd.read_csv(raw_path)
    return data


def data_preparation(data, printing = PRINTING_STATE):
    '''
    Data preparation step.

    Input:
        data -      The raw data csv, as read by data_extraction.
        printing -  Set to True to enable printing
    
    Output:

    '''
    # We set the dates for the data we want.
    max_date = "2024-01-31"
    min_date = "2024-01-01"

    # If not set, max_date is set to the present day.
    if not max_date:
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()

    min_date = pd.to_datetime(min_date).date()

    # Apply time limit to data
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    # We update the min and max dates to correspond to dates present in the dataset. For good measure, we dump the date limits in a json.
    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}

    date_limit_path = INTERIM_DATA_DIR / "date_limits.json"
    with open(date_limit_path, "w") as f:
        json.dump(date_limits, f)

    data = data.drop(
        [
            "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"
        ],
        axis=1
    )   

    #Removing columns that will be added back after the EDA (MATTI: Don't think this is ever used again..? )
    data = data.drop(
        ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
        axis=1
    )


    #Remove rows with empty target variable
    #Remove rows with other invalid column data
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    data = data[data.source == "signup"]
    result=data.lead_indicator.value_counts(normalize = True)

    if printing:
        print("Target value counter")
        for val, n in zip(result.index, result):
            print(val, ": ", n)
        print(data)


    vars = [
        "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
    ]

    for col in vars:
        data[col] = data[col].astype("object")
        if printing:
           print(f"Changed {col} to object type")

    
    # Separate categorical and continuous columns
    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]

    if printing:
        print("\nContinuous columns: \n")
        pprint(list(cont_vars.columns), indent=4)
        print("\n Categorical columns: \n")
        pprint(list(cat_vars.columns), indent=4)


    ## NOW TO HANDLE THE OUTLIERS/NaN
    # Outliers:
    cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                             upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary_path = INTERIM_DATA_DIR / "outlier_summary.csv"
    outlier_summary.to_csv(outlier_summary_path)
    if printing:
        outlier_summary

    # Missing values:
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute_path = INTERIM_DATA_DIR / "cat_missing_impute.csv"
    cat_missing_impute.to_csv(cat_missing_impute_path)
    if printing:
        cat_missing_impute

    # Continuous variables missing values using helper functions
    cont_vars = cont_vars.apply(impute_missing_values)
    cont_vars.apply(describe_numeric_col).T
    if printing:
        cont_vars

    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T
    if printing:
        cat_vars


    ## Added a scaler for data standardisation
    # We save this in the models directory
    scaler_path = MODELS_DIR / "scaler.pkl"

    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    joblib.dump(value=scaler, filename=scaler_path)
    if printing:
        print("Saved scaler in artifacts")

    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    if printing:
        cont_vars

    # Now we combine the data into one final dataframe
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    if printing:
        print(f"Data cleansed and combined.\nRows: {len(data)}")
        data

    ## We add the artifact drift columns for future use
    data_columns = list(data.columns)
    data_columns_path = INTERIM_DATA_DIR / "columns_drift.json"
    with open(data_columns_path,'w+') as f:           
        json.dump(data_columns,f)
        
    # And then save the training data we have got so far after the first filtering
    training_data_path = INTERIM_DATA_DIR / "training_data.csv"
    data.to_csv(training_data_path, index=False)

    if printing:
        data.columns

    # We bin based on the Source column (But all the data in our dataset is 'signup')
    data['bin_source'] = data['source']
    values_list = ['li', 'organic','signup','fb']
    data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
    mapping = {'li' : 'socials', 
            'fb' : 'socials', 
            'organic': 'group1', 
            'signup': 'group1'
            }

    data['bin_source'] = data['source'].map(mapping)
    


    return data
