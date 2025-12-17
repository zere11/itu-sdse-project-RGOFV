import os
import pandas as pd
import datetime
import json
import warnings
import numpy as np
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
import joblib

from helpers import describe_numeric_col, impute_missing_values


def data_preparation(data, printing=False):
    # ENTIRE original code from 01_data.py features section
    max_date = "2024-01-31"
    min_date = "2024-01-01"

    if not max_date:
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()
    min_date = pd.to_datetime(min_date).date()

    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    with open("./artifacts/date_limits.json", "w") as f:
        json.dump(date_limits, f)

    cols_to_drop = ["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"]
    cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    if cols_to_drop:
        data = data.drop(cols_to_drop, axis=1)

    cols_to_drop2 = ["domain", "country", "visited_learn_more_before_booking", "visited_faq"]
    cols_to_drop2 = [col for col in cols_to_drop2 if col in data.columns]
    if cols_to_drop2:
        data = data.drop(cols_to_drop2, axis=1)

    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    data = data[data.source == "signup"]
    result = data.lead_indicator.value_counts(normalize=True)

    if printing:
        print("Target value counter")
        for val, n in zip(result.index, result):
            print(val, ": ", n)
        print(data)

    vars = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]

    for col in vars:
        data[col] = data[col].astype("object")
        if printing:
            print(f"Changed {col} to object type")

    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]

    if printing:
        print("\nContinuous columns: \n")
        pprint(list(cont_vars.columns), indent=4)
        print("\n Categorical columns: \n")
        pprint(list(cat_vars.columns), indent=4)

    cont_vars = cont_vars.apply(lambda x: x.clip(lower=(x.mean()-2*x.std()), upper=(x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv('./artifacts/outlier_summary.csv')

    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv("./artifacts/cat_missing_impute.csv")

    cont_vars = cont_vars.apply(impute_missing_values)
    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)

    scaler_path = "./artifacts/scaler.pkl"
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    joblib.dump(value=scaler, filename=scaler_path)

    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)

    data_columns = list(data.columns)
    with open('./artifacts/columns_drift.json','w+') as f:
        json.dump(data_columns,f)

    data.to_csv('./artifacts/training_data.csv', index=False)

    data['bin_source'] = data['source']
    values_list = ['li', 'organic','signup','fb']
    data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
    mapping = {'li':'socials','fb':'socials','organic':'group1','signup':'group1'}
    data['bin_source'] = data['source'].map(mapping)

    return data
