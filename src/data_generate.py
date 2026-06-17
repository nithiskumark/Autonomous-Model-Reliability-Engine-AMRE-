"""
Autonoumous Model Reliability Engine (AMRE)
Production ML Monitoring Engine

Data Generator

Creates synthetic production-grade dataset and
simulates real-world data streams for model
monitoring, drift detection, reliability analysis,
root-cause diagnosis, and autonomous recovery.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# DATA GENERATION
def gen_data(n=5000):
    
    np.random.seed(42)

    # Generate synthetic customer financial data

    age = np.random.randint(19, 60, n).round()

    base_income = 8000

    age_income_effect = -10 * (age - 45) ** 2 + 55000

    raw_income = (
        base_income
        + age_income_effect
        + np.random.lognormal(9, 0.5, n)
    )

    income = (
        (raw_income - raw_income.mean())
        / raw_income.std()
    )

    income = (
        income * 20000 + 60000
    ).round(2)

    savings_rate = np.random.uniform(
        0.05,
        0.30,
        n
    )

    debt_ratio = np.random.uniform(
        0.10,
        0.60,
        n
    )

    noise = np.random.normal(
        0,
        0.03,
        n
    )

    age_purchase_effect = (
        -10 * (age - 40) ** 2 + 500
    )

    age_purchase_effect = (
        age_purchase_effect
        - age_purchase_effect.mean()
    ) / age_purchase_effect.std()

    purchase_power = (
        income * (
            1
            - savings_rate
            - debt_ratio
        )
        + age_purchase_effect * 5000
        + noise * 4000
    ).round(2)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "savings_rate": savings_rate.round(2),
            "debt_ratio": debt_ratio.round(2),
            "purchase_power": purchase_power,
        }
    )

    scaler = MinMaxScaler(
        feature_range = (1, 100)
    )

    purchasing_power = scaler.fit_transform(
        df[["purchase_power"]]
    )

    df["purchasing_power"] = pd.DataFrame(
        np.round(purchasing_power, 2), 
        columns=["purchasing_power"]
    )

    df.to_csv(
        "data/financial_data.csv",
        index = False
    )

    print(
        "Data File Saved"
    )

    return df