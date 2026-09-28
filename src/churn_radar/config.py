from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", MODEL_DIR / "churn_model.joblib"))

RANDOM_STATE = 42
TEST_SIZE = 0.2

HIGH_RISK = 0.60
MEDIUM_RISK = 0.35

# Business-impact simulation: share of contacted churners retained for a year.
ASSUMED_SAVE_RATE = 0.30
TARGET_SHARE = 0.20
