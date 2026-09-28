"""Train, compare, and persist the churn model.

python -m churn_radar.train            # train and write artifacts
python -m churn_radar.train --mlflow   # also log the run to MLflow
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from churn_radar import config  # noqa: E402
from churn_radar.features import (  # noqa: E402
    TARGET,
    TENURE_LABELS,
    build_preprocessor,
    engineer_features,
    load_and_clean,
)

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def candidate_models(pos_weight: float) -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return ModelResult(
        name=name,
        accuracy=round(accuracy_score(y_test, pred), 4),
        precision=round(precision_score(y_test, pred), 4),
        recall=round(recall_score(y_test, pred), 4),
        f1=round(f1_score(y_test, pred), 4),
        roc_auc=round(roc_auc_score(y_test, proba), 4),
    )


def save_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    df.groupby("Contract")[TARGET].mean().sort_values().plot.barh(ax=axes[0], color="#4C6EF5")
    axes[0].set_title("Churn rate by contract type")
    axes[0].set_xlabel("Churn rate")

    df.groupby("tenure_bucket")[TARGET].mean().reindex(TENURE_LABELS).plot.bar(
        ax=axes[1], color="#F76707", rot=0
    )
    axes[1].set_title("Churn rate by tenure")

    df.boxplot(column="MonthlyCharges", by=TARGET, ax=axes[2])
    axes[2].set_title("Monthly charges vs churn")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "eda_overview.png", dpi=150)
    plt.close(fig)


def business_impact(X_test: pd.DataFrame, y_test: pd.Series, proba) -> dict:
    """Estimate revenue protected by calling only the riskiest customers."""
    scored = X_test.assign(churn_prob=proba, actual=y_test.to_numpy())
    targeted = scored.nlargest(int(len(scored) * config.TARGET_SHARE), "churn_prob")
    caught = int(targeted["actual"].sum())
    avg_monthly = scored.loc[scored["actual"] == 1, "MonthlyCharges"].mean()
    saved = caught * config.ASSUMED_SAVE_RATE * avg_monthly * 12
    return {
        "top20pct_capture_rate": round(caught / scored["actual"].sum(), 4),
        "churners_caught_in_top20pct": caught,
        "assumed_save_rate": config.ASSUMED_SAVE_RATE,
        "estimated_annual_revenue_saved_usd": round(float(saved), 2),
    }


def log_to_mlflow(results: list[ModelResult], best: str, pipe: Pipeline, metrics: dict) -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise SystemExit("--mlflow needs MLflow: pip install mlflow") from exc

    mlflow.set_experiment("churn-radar")
    for result in results:
        with mlflow.start_run(run_name=result.name):
            mlflow.log_params({"model": result.name, "test_size": config.TEST_SIZE})
            mlflow.log_metrics({k: v for k, v in asdict(result).items() if k != "name"})
            if result.name == best:
                mlflow.set_tag("best_model", "true")
                mlflow.log_metrics(
                    {
                        "top20pct_capture_rate": metrics["top20pct_capture_rate"],
                        "est_annual_revenue_saved_usd": metrics[
                            "estimated_annual_revenue_saved_usd"
                        ],
                    }
                )
                mlflow.sklearn.log_model(pipe, name="model")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow", action="store_true", help="log runs to MLflow")
    args = parser.parse_args(argv)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = engineer_features(load_and_clean(config.DATA_PATH))
    save_eda(df)
    print(f"[data] {len(df):,} customers, churn rate {df[TARGET].mean():.1%}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    results: list[ModelResult] = []
    fitted: dict[str, Pipeline] = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, clf in candidate_models(pos_weight).items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        result = evaluate(name, pipe, X_test, y_test)
        results.append(result)
        fitted[name] = pipe
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
        print(f"[model] {name}: roc_auc={result.roc_auc} recall={result.recall}")

    ax.set_title("ROC curves, churn models (test set)")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    best = max(results, key=lambda r: r.roc_auc)
    best_pipe = fitted[best.name]
    joblib.dump(best_pipe, config.MODEL_PATH)

    proba = best_pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "dataset_rows": int(len(df)),
        "churn_rate": round(float(y.mean()), 4),
        "models": [asdict(r) for r in results],
        "best_model": best.name,
        "confusion_matrix": confusion_matrix(y_test, (proba >= 0.5).astype(int)).tolist(),
        **business_impact(X_test, y_test, proba),
    }
    (config.OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[best] {best.name} saved to {config.MODEL_PATH}")

    if args.mlflow:
        log_to_mlflow(results, best.name, best_pipe, metrics)
    return metrics


if __name__ == "__main__":
    main()
