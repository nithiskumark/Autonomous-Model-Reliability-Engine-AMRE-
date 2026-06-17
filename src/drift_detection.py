"""
Autonoumous Model Reliability Engine (AMRE)
Detect the drift in the data
"""

import numpy as np
import pandas as pd
import json
from scipy.stats import ks_2samp

def data_quality_score(df):
    """
    Calculates overall data quality score.

    Returns:
        score (0-100)
    """

    numeric_df = df.select_dtypes(include=np.number)

    missing_rate = numeric_df.isnull().mean().mean()

    z_scores = np.abs(
        (numeric_df - numeric_df.mean()) /
        numeric_df.std(ddof=0)
    )

    outlier_rate = (z_scores > 3).mean().mean()

    score = 100 - (missing_rate * 50 + outlier_rate * 50)

    return round(max(score, 0), 2)

def calculate_psi(expected, actual, buckets=10):

    """
    Calculates Population Stability Index.

    PSI Value  ---- Meaning
    < 0.1      ---- No significant drift
    0.1 - 0.25 ---- Moderate drift
    > 0.25	   ---- Significant drift
    """

    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.percentile(
        expected,
        np.arange(0, 101, 100 / buckets)
    )

    expected_counts = np.histogram(
        expected,
        bins=breakpoints
    )[0]

    actual_counts = np.histogram(
        actual,
        bins=breakpoints
    )[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    expected_perc = np.where(
        expected_perc == 0,
        0.0001,
        expected_perc
    )

    actual_perc = np.where(
        actual_perc == 0,
        0.0001,
        actual_perc
    )

    psi = np.sum(
        (expected_perc - actual_perc)
        *
        np.log(expected_perc / actual_perc)
    )

    return round(psi, 4)

def feature_drift_report(
        reference_df,
        current_df,
        psi_threshold=0.2):

    numeric_cols = reference_df.select_dtypes(
        include=np.number
    ).columns

    drift_results = []

    for col in numeric_cols:

        psi = calculate_psi(
            reference_df[col].dropna(),
            current_df[col].dropna()
        )

        drift_results.append({
            "feature": col,
            "psi": psi,
            "drift_detected": psi > psi_threshold
        })

    return pd.DataFrame(drift_results)

def ks_drift_report(
        reference_df,
        current_df,
        p_threshold=0.05):

    numeric_cols = reference_df.select_dtypes(
        include=np.number
    ).columns

    results = []

    for col in numeric_cols:

        statistic, p_value = ks_2samp(
            reference_df[col].dropna(),
            current_df[col].dropna()
        )

        results.append({
            "feature": col,
            "ks_statistic": round(statistic, 4),
            "p_value": round(p_value, 4),
            "drift_detected": p_value < p_threshold
        })

    return pd.DataFrame(results)

def overall_drift_score(drift_report):

    drift_rate = (
        drift_report["drift_detected"]
        .mean()
    )

    score = 100 - drift_rate * 100

    return round(score, 2)

def reliability_score(
        quality_score,
        drift_score):

    reliability = (
        quality_score * 0.5 +
        drift_score * 0.5
    )

    return round(reliability, 2)

def run_amre_check(
        reference_df,
        current_df):

    quality = data_quality_score(current_df)

    psi_report = feature_drift_report(
        reference_df,
        current_df
    )

    psi = pd.DataFrame(psi_report)

    psi.to_csv(
        "artifacts/psi_report.csv", 
        index=False
    )

    drift_score = overall_drift_score(
        psi_report
    )

    reliability = reliability_score(
        quality,
        drift_score
    )

    psi_report = psi_report.to_dict(orient="records")

    result = {
        "quality_score": quality,
        "drift_score": drift_score,
        "reliability_score": reliability,
        "psi_report": psi_report
    }

    with open("artifacts/drift_report.json", "w") as f:
        json.dump(
            result,
            f,
            indent=4
        )

    return result

if __name__ == "__main__":

    df = pd.read_csv("data/financial_data.csv")

    df_d = pd.read_csv("data/drifted_data.csv")

    result = run_amre_check(
        df,
        df_d
    )

    print("\nQUALITY SCORE")
    print(result["quality_score"])

    print("\nDRIFT SCORE")
    print(result["drift_score"])

    print("\nRELIABILITY SCORE")
    print(result["reliability_score"])

    print("\nPSI REPORT")
    print(result["psi_report"])


