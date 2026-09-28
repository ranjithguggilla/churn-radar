from __future__ import annotations

import json

import joblib

from churn_radar import config, train


def test_training_writes_model_metrics_and_charts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "churn_model.joblib")

    metrics = train.main([])

    assert {m["name"] for m in metrics["models"]} == {
        "LogisticRegression",
        "RandomForest",
        "XGBoost",
    }
    assert all(m["roc_auc"] > 0.8 for m in metrics["models"])
    assert metrics["top20pct_capture_rate"] > 0.4
    assert json.loads((config.OUTPUT_DIR / "metrics.json").read_text()) == metrics
    assert (config.OUTPUT_DIR / "roc_curves.png").exists()
    assert (config.OUTPUT_DIR / "eda_overview.png").exists()
    assert hasattr(joblib.load(config.MODEL_PATH), "predict_proba")
