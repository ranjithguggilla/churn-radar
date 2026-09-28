"""FastAPI scoring service.

uvicorn churn_radar.api:app --reload    # docs at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import Depends, FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from churn_radar import __version__, config
from churn_radar.explain import recommended_action, risk_drivers, risk_tier
from churn_radar.features import customers_to_frame
from churn_radar.schemas import Customer, Health, Prediction

MAX_BATCH = 1000

app = FastAPI(
    title="ChurnRadar API",
    version=__version__,
    description="Score telecom customers for churn risk and get a retention action.",
)


@lru_cache(maxsize=1)
def _load(path: Path) -> Pipeline:
    return joblib.load(path)


def get_model() -> Pipeline:
    if not config.MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="model not trained yet: run `python -m churn_radar.train`",
        )
    return _load(config.MODEL_PATH)


Model = Annotated[Pipeline, Depends(get_model)]


def _score(model: Pipeline, customers: list[Customer]) -> list[Prediction]:
    records = [c.model_dump() for c in customers]
    probabilities = model.predict_proba(customers_to_frame(records))[:, 1]
    return [
        Prediction(
            churn_probability=round(float(p), 4),
            risk_tier=risk_tier(p),
            risk_drivers=risk_drivers(record),
            recommended_action=recommended_action(p),
        )
        for record, p in zip(records, probabilities, strict=True)
    ]


@app.get("/health", response_model=Health)
def health() -> Health:
    loaded = config.MODEL_PATH.exists()
    return Health(status="ok" if loaded else "degraded", model_loaded=loaded, version=__version__)


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, model: Model) -> Prediction:
    return _score(model, [customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer], model: Model) -> list[Prediction]:
    if not customers:
        raise HTTPException(status_code=422, detail="send at least one customer")
    if len(customers) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"batch limit is {MAX_BATCH} customers")
    return _score(model, customers)
