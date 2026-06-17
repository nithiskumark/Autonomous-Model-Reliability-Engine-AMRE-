"""
Autonoumous Model Reliability Engine (AMRE)
Decision Engine
"""\

import pandas as pd
import json

from baseline_model_train import (
    prep_data,
    train_model,
    evaluate_model
)

def cal_drift_severity(psi_report):

    psi_values = psi_report["psi"].tolist()

    max_psi = max(psi_values)

    drifted_features = sum(
        psi_report["drift_detected"]
    )

    drift_severity = {
        "max_psi": max_psi,
        "drifted_features": drifted_features
    }

    return drift_severity

def model_health_score(
    r2,
    rmse,
    mae
):

    score = 0

    # R²

    if r2 > 0.9:
        score += 40

    elif r2 > 0.8:
        score += 30

    elif r2 > 0.7:
        score += 20

    # RMSE

    if rmse < 5:
        score += 30

    elif rmse < 10:
        score += 20

    # MAE

    if mae < 5:
        score += 30

    elif mae < 10:
        score += 20

    return score

def decision_engine(
    quality_score,
    reliability_score,
    psi_report,
    r2,
    rmse,
    mae
):

    drift_info = cal_drift_severity(
        psi_report
    )

    model_score = model_health_score(
        r2,
        rmse,
        mae
    )

    max_psi = drift_info["max_psi"]

    drifted_features = drift_info[
        "drifted_features"
    ]

    # Decision Rules

    if quality_score < 70:

        action = "BLOCK_RETRAIN"

    elif max_psi > 3:

        action = "RETRAIN_FULL_MODEL"

    elif drifted_features >= 3:

        action = "RETRAIN_FULL_MODEL"

    elif reliability_score < 60:

        action = "RETRAIN_FULL_MODEL"

    elif model_score < 60:

        action = "RETRAIN_FULL_MODEL"

    elif max_psi > 1:

        action = "RECALIBRATE"

    else:

        action = "MONITOR"

    decision = {
        "action": action,
        "model_score": model_score,
        "max_psi": max_psi,
        "drifted_features": drifted_features
    }

    return decision


if __name__ == "__main__":

    psi_report = pd.read_csv("artifacts/psi_report.csv")

    drift_severity = cal_drift_severity(psi_report)

    with open(
        "artifacts/model_metrics_drift.json",
        "r",
    ) as f:
        metrics = json.load(f)

    r2 = metrics["r2"]
    mae = metrics["mae"]
    rmse = metrics["rmse"]

    score = model_health_score(r2, mae, rmse)

    with open(
        "artifacts/drift_report.json",
        "r",
    ) as f:
        report = json.load(f)

    quality_score = report["quality_score"]
    reliability_score = report["reliability_score"]

    decision = decision_engine(
        quality_score,
        reliability_score,
        psi_report,
        r2,
        rmse,
        mae
    )

    print(decision)

    if decision["action"] == "RETRAIN_FULL_MODEL":

        df = pd.read_csv("data/drifted_data.csv")

        X_train, X_test, y_train, y_test = prep_data(df)

        model = train_model(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        print(f"Metrics of the retrained model: {metrics}")

        













