"""Gradio front-end for a model bundle produced by train.py."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import pandas as pd

from predict import prepare_features

MODEL_PATH = Path(os.environ.get("VPN_MODEL_PATH", "models/xgb_15s.joblib"))


def load_bundle(path: Path) -> tuple[Any, dict]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "metadata" not in bundle:
        raise ValueError(
            f"{path} is not a model bundle produced by train.py."
        )
    return bundle["pipeline"], bundle["metadata"]


try:
    PIPELINE, METADATA = load_bundle(MODEL_PATH)
except FileNotFoundError as exc:
    raise SystemExit(
        f"Model not found at {MODEL_PATH}. Run `python train.py --window 15` first."
    ) from exc


def predict_csv(file_path: str | None):
    if not file_path:
        return {"error": "No CSV file selected."}, pd.DataFrame()

    try:
        frame = pd.read_csv(file_path)
        features = prepare_features(frame, METADATA)
        predictions = PIPELINE.predict(features).astype(int)
        probabilities = PIPELINE.predict_proba(features)[:, 1]
    except Exception as exc:
        return {"error": str(exc)}, pd.DataFrame()

    output = frame.copy()
    output["prediction"] = predictions
    output["vpn_probability"] = probabilities

    vpn_count = int((predictions == 1).sum())
    summary = {
        "rows": int(len(output)),
        "vpn": vpn_count,
        "non_vpn": int(len(output) - vpn_count),
        "window_seconds": METADATA.get("window_seconds"),
    }
    return summary, output


with gr.Blocks(title="VPN Traffic Classifier") as demo:
    gr.Markdown(
        """
# VPN Traffic Classifier

Upload a CSV containing the same flow-feature schema used to train the model.
The output includes the binary class and the estimated probability of VPN traffic.

This is a research demo trained on ISCXVPN2016 data; it is not a universal VPN detector.
"""
    )
    input_file = gr.File(label="Flow CSV", file_types=[".csv"], type="filepath")
    predict_button = gr.Button("Run prediction", variant="primary")
    output_summary = gr.JSON(label="Summary")
    output_table = gr.Dataframe(label="Predictions")

    predict_button.click(
        predict_csv,
        inputs=input_file,
        outputs=[output_summary, output_table],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
