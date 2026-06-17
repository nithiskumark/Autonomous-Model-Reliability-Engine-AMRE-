"""
Autonoumous Model Reliability Engine (AMRE)
Train the Baseline model
"""

import os
import joblib
import json
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_generate import gen_data

def prep_data(df):

    cols = ["purchase_power", "purchasing_power"]

    X = df.drop(
        columns=cols, 
        axis=1
    )

    y = df["purchasing_power"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.9,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):

    """
    Train baseline Linear Regression model
    and save it to artifacts folder.
    """

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    os.makedirs(
        "artifacts", 
        exist_ok=True
    )

    model_path = "artifacts/baseline_model.pkl"

    joblib.dump(
        model,
        model_path
    )
    
    print(f"Model saved at: {model_path}")

    return model

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

    with open("artifacts/model_metrics.json", "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    return metrics

if __name__ == "__main__":

    # LOAD DATA
    df = gen_data()
    
    X_train, X_test, y_train, y_test = prep_data(df)

    # TRAIN MODEL
    model = train_model(X_train, y_train)

    # EVALUATE THE MODEL
    metrics = evaluate_model(model, X_test, y_test)

    print(metrics)


























