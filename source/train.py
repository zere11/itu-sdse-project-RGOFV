import pandas as pd
import os
import datetime
import json
import warnings
import numpy as np
from pprint import pprint
import shutil
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    f1_score
)
from xgboost import XGBRFClassifier
from scipy.stats import uniform, randint
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

from utils.helpers import create_dummy_cols
from wrappers.mlflow_wrapper import lr_wrapper

def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
          name=model_name,
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)

def load_and_prepare_data(data_gold_path, printing=False):
    data = pd.read_csv(data_gold_path)
    if printing:
        print(f"Training data length: {len(data)}")
        print(data.head(5))

    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]

    other_vars = data.drop(cat_cols, axis=1)

    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data:
        data[col] = data[col].astype("float64")
        if printing:
            print(f"Changed column {col} to float")

    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )

    if printing:
        print(y_train)

    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, X_test, y_train, y_test, printing=False):
    model = XGBRFClassifier(random_state=42)
    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }

    model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)
    model_grid.fit(X_train, y_train)

    best_model_xgboost_params = model_grid.best_params_
    if printing:
        print("Best xgboost params")
        pprint(best_model_xgboost_params)

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    if printing:
        print("Accuracy train", accuracy_score(y_pred_train, y_train))
        print("Accuracy test", accuracy_score(y_pred_test, y_test))
        print("Test actual/predicted\n")
        print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
        print("Classification report\n")
        print(classification_report(y_test, y_pred_test), '\n')
        print("Train actual/predicted\n")
        print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
        print("Classification report\n")
        print(classification_report(y_train, y_pred_train), '\n')

    xgboost_model = model_grid.best_estimator_
    xgboost_model_path = "./artifacts/lead_model_xgboost.json"
    xgboost_model.save_model(xgboost_model_path)

    model_results = {
        xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
    }

    return xgboost_model_path, model_results, xgboost_model


def train_logistic_regression_model(X_train, X_test, y_train, y_test, experiment_name, data_version, printing=False):
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression()
        lr_model_path = "./artifacts/lead_model_lr.pkl"

        params = {
            'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            'penalty': ["none", "l1", "l2", "elasticnet"],
            'C': [100, 10, 1.0, 0.1, 0.01]
        }
        model_grid = RandomizedSearchCV(model, param_distributions=params, verbose=3, n_iter=10, cv=3)
        model_grid.fit(X_train, y_train)

        best_model = model_grid.best_estimator_
        y_pred_train = model_grid.predict(X_train)
        y_pred_test = model_grid.predict(X_test)

        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", data_version)
        
        joblib.dump(value=best_model, filename=lr_model_path)
        mlflow.pyfunc.log_model('model', python_model=lr_wrapper(best_model))

    model_classification_report = classification_r_
