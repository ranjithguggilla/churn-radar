from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn_radar import api, config


@pytest.fixture
def client(model_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    return TestClient(api.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.joblib")
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "degraded"
    assert client.post("/predict", json={}).status_code in (422, 503)


def test_predict_ranks_risky_customer_above_loyal_one(client, customer, loyal_customer):
    risky = client.post("/predict", json=customer)
    loyal = client.post("/predict", json=loyal_customer)
    assert risky.status_code == 200 and loyal.status_code == 200
    risky, loyal = risky.json(), loyal.json()
    assert 0 <= loyal["churn_probability"] < risky["churn_probability"] <= 1
    assert risky["risk_tier"] == "high"
    assert loyal["risk_tier"] == "low"
    assert loyal["risk_drivers"] == []


def test_predict_rejects_invalid_values(client, customer):
    assert client.post("/predict", json={**customer, "Contract": "Weekly"}).status_code == 422
    assert client.post("/predict", json={**customer, "tenure": -1}).status_code == 422
    assert client.post("/predict", json={**customer, "unknown": 1}).status_code == 422


def test_batch(client, customer, loyal_customer):
    response = client.post("/predict/batch", json=[customer, loyal_customer])
    assert response.status_code == 200
    assert [p["risk_tier"] for p in response.json()] == ["high", "low"]
    assert client.post("/predict/batch", json=[]).status_code == 422


def test_batch_limit(client, customer, monkeypatch):
    monkeypatch.setattr(api, "MAX_BATCH", 2)
    assert client.post("/predict/batch", json=[customer] * 3).status_code == 413
