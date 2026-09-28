from __future__ import annotations

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_radar import config
from churn_radar.features import TARGET, build_preprocessor, engineer_features, load_and_clean


@pytest.fixture(scope="session")
def clean_df():
    return engineer_features(load_and_clean(config.DATA_PATH))


@pytest.fixture(scope="session")
def model_path(clean_df, tmp_path_factory):
    """A small logistic model trained on the real data, so tests never need train.py."""
    X = clean_df.drop(columns=[TARGET])
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(X)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipe.fit(X, clean_df[TARGET])
    path = tmp_path_factory.mktemp("models") / "churn_model.joblib"
    joblib.dump(pipe, path)
    return path


@pytest.fixture
def customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
    }


@pytest.fixture
def loyal_customer(customer):
    return {
        **customer,
        "tenure": 70,
        "Partner": "Yes",
        "Dependents": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "Contract": "Two year",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 60.0,
    }
