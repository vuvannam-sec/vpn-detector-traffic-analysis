"""Batch inference for CSV flow records."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score flow records with a trained model.")
    parser.add_argument("--model", type=Path, required=True, help="Model bundle created by train.py.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV containing flow features.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    return parser.parse_args()


def load_bundle(path: Path) -> tuple[Any, dict]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "metadata" not in bundle:
        raise ValueError(
            "Unsupported model artifact. Expected a bundle produced by train.py."
        )
    return bundle["pipeline"], bundle["metadata"]


def prepare_features(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    expected = metadata.get("feature_columns")
    if not expected:
        raise ValueError("Model metadata does not contain feature_columns.")

    working = frame.drop(columns=["label"], errors="ignore")
    missing = [column for column in expected if column not in working.columns]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(f"Input is missing {len(missing)} required column(s): {preview}{suffix}")

    return working.loc[:, expected]


def score_frame(frame: pd.DataFrame, pipeline: Any, metadata: dict) -> pd.DataFrame:
    features = prepare_features(frame, metadata)
    predictions = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)[:, 1]

    result = frame.copy()
    result["prediction"] = predictions.astype(int)
    result["vpn_probability"] = probabilities
    return result


def main() -> None:
    args = parse_args()

    pipeline, metadata = load_bundle(args.model)
    frame = pd.read_csv(args.input)
    result = score_frame(frame, pipeline, metadata)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    vpn_count = int((result["prediction"] == 1).sum())
    print(f"Wrote {len(result)} rows to {args.output}")
    print(f"VPN={vpn_count} Non-VPN={len(result) - vpn_count}")


if __name__ == "__main__":
    main()
