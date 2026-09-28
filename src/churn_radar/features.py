"""Cleaning and feature engineering shared by training, the API, and the UI.

Keeping one implementation here means a customer scored at inference time goes
through exactly the same transformations as the rows the model was trained on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"

TENURE_BINS = [-1, 6, 12, 24, 48, 100]
TENURE_LABELS = ["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
NO_SERVICE = {"No", "No internet service", "No phone service"}

RAW_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    *SERVICE_COLUMNS,
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load the IBM Telco churn CSV and fix its known quirks."""
    df = pd.read_csv(path)
    # Customers in their first month have a blank TotalCharges string.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df = df.drop(columns=["customerID"])
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived features the model is trained on."""
    df = df.copy()
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    tenure = df["tenure"].to_numpy(dtype=float)
    df["avg_monthly_spend"] = np.where(
        tenure > 0,
        df["TotalCharges"] / np.where(tenure > 0, tenure, 1.0),
        df["MonthlyCharges"],
    )
    df["num_services"] = (~df[SERVICE_COLUMNS].isin(NO_SERVICE)).sum(axis=1)
    return df


def customers_to_frame(customers: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Turn raw customer records into a model-ready frame.

    TotalCharges is optional; when absent it is estimated as MonthlyCharges x
    tenure, which is how the billing system accrues it.
    """
    rows = []
    for customer in customers:
        row = {col: customer.get(col) for col in RAW_COLUMNS}
        if row["TotalCharges"] is None:
            row["TotalCharges"] = float(row["MonthlyCharges"]) * int(row["tenure"])
        rows.append(row)
    return engineer_features(pd.DataFrame(rows, columns=RAW_COLUMNS))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
