"""Train a VPN/non-VPN classifier from the cleaned ISCXVPN2016 flow tables."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

SUPPORTED_WINDOWS = (15, 30, 60, 120)
DEFAULT_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost VPN/non-VPN classifier from cleaned flow CSV files."
    )
    parser.add_argument(
        "--window",
        type=int,
        choices=SUPPORTED_WINDOWS,
        default=15,
        help="Flow aggregation window in seconds (default: 15).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("clean_data"),
        help="Directory containing vpn{window}s_cleaned.csv and novpn{window}s_cleaned.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Model bundle output path (default: models/xgb_<window>s.joblib).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Metrics JSON path (default: alongside model as *.metrics.json).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Fraction used for the stratified hold-out test set (default: 0.30).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Optional stratified cross-validation folds. Use 0 to skip (default: 0).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for split and model training.",
    )
    return parser.parse_args()


def load_labeled_data(data_dir: Path, window: int) -> tuple[pd.DataFrame, pd.Series]:
    vpn_path = data_dir / f"vpn{window}s_cleaned.csv"
    nonvpn_path = data_dir / f"novpn{window}s_cleaned.csv"

    missing_files = [str(path) for path in (vpn_path, nonvpn_path) if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required dataset file(s): " + ", ".join(missing_files)
        )

    vpn = pd.read_csv(vpn_path)
    nonvpn = pd.read_csv(nonvpn_path)

    if list(vpn.columns) != list(nonvpn.columns):
        raise ValueError(
            "VPN and non-VPN datasets do not have the same feature columns/order."
        )

    vpn = vpn.copy()
    nonvpn = nonvpn.copy()
    vpn["label"] = 1
    nonvpn["label"] = 0

    combined = pd.concat([vpn, nonvpn], ignore_index=True)
    return combined.drop(columns=["label"]), combined["label"].astype(int)


def build_pipeline(X: pd.DataFrame, random_state: int) -> Pipeline:
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(transformers=transformers)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_holdout(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    random_state: int,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("--cv-folds must be 0 or at least 2.")

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        n_jobs=1,
        return_train_score=False,
    )

    summary: dict[str, Any] = {"folds": folds}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        values = scores[f"test_{metric}"]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_fold": [float(value) for value in values],
        }
    return summary


def metrics_path_for(model_path: Path) -> Path:
    return model_path.with_suffix(".metrics.json")


def main() -> None:
    args = parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")

    model_path = args.output or Path("models") / f"xgb_{args.window}s.joblib"
    metrics_path = args.metrics_output or metrics_path_for(model_path)

    X, y = load_labeled_data(args.data_dir, args.window)
    pipeline = build_pipeline(X, args.random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    pipeline.fit(X_train, y_train)
    holdout_metrics = evaluate_holdout(pipeline, X_test, y_test)

    run_metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_seconds": args.window,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "rows_total": int(len(X)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "positive_rows": int(y.sum()),
        "negative_rows": int((y == 0).sum()),
        "feature_columns": list(X.columns),
        "holdout": holdout_metrics,
    }

    if args.cv_folds:
        run_metadata["cross_validation"] = evaluate_cross_validation(
            build_pipeline(X, args.random_state),
            X,
            y,
            args.cv_folds,
            args.random_state,
        )

    bundle = {
        "pipeline": pipeline,
        "metadata": run_metadata,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, model_path)
    metrics_path.write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")
    print(
        "Hold-out: "
        f"F1={holdout_metrics['f1']:.4f} "
        f"ROC-AUC={holdout_metrics['roc_auc']:.4f} "
        f"Accuracy={holdout_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
