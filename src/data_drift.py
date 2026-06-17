"""
Autonoumous Model Reliability Engine (AMRE)

Drift Data Generator

Creates synthetic production-grade drifted dataset and
simulates real-world data streams for model
monitoring, drift detection, reliability analysis,
root-cause diagnosis, and autonomous recovery.
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def gen_drift_data(n=5000):

    np.random.seed(42)

    df = pd.read_csv("data/financial_data.csv")

    income = df["income"]
    age = df["age"]
    savings_rate = df["savings_rate"]
    debt_ratio = df["debt_ratio"]


    income += np.random.normal(
        60000,
        12000,
        n
    )

    age += np.random.randint(
        15,
        25,
        n
    )

    savings_rate += np.random.normal(
        0.25,
        0.05,
        n
    )

    debt_ratio += np.random.normal(
        0.30,
        0.03,
        n
    )

    savings_rate = np.clip(
        savings_rate,
        0,
        1
    )

    debt_ratio = np.clip(
        debt_ratio,
        0,
        1
    )

    age_effect = -10 * (age - 45)**2 + 55000

    noise = np.random.normal(0,0.03,n)

    purchase_power = (
        (income * (1 - savings_rate - debt_ratio))
        + (age_effect * 5000)
        + (noise * 4000)
    ).round(2)

    data_d = {
        "age": age,
        "income": income.round(2),
        "savings_rate": savings_rate.round(2),
        "debt_ratio": debt_ratio.round(2),
        "purchase_power": purchase_power
    }

    df_d = pd.DataFrame(data_d)

    scaler = MinMaxScaler(
        feature_range=(1, 100)
    )

    purchasing_power = scaler.fit_transform(
        df_d[["purchase_power"]]
    )

    df_d["purchasing_power"] = (
        purchasing_power
        .round(2)
    )

    df_d.to_csv(
        "data/drifted_data.csv",
        index=False
    )

    return df_d

def evaluate_model_for_drift(df_d):

    cols = ["purchase_power", "purchasing_power"]

    X = df_d.drop(
        columns=cols, 
        axis=1
    )

    y = df_d["purchasing_power"]

    model = joblib.load("artifacts/baseline_model.pkl")

    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y, y_pred)

    metrics_drift = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

    with open("artifacts/model_metrics_drift.json", "w") as f:
        json.dump(
            metrics_drift,
            f,
            indent=4
        )

    return metrics_drift

if __name__ == "__main__":

    df_d = gen_drift_data()

    drift_metrics = evaluate_model_for_drift(df_d)





