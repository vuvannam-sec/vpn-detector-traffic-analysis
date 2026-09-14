from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from predict import load_bundle, prepare_features, score_frame
from train import build_pipeline, load_labeled_data


def test_real_15s_data_can_train_and_score(tmp_path: Path) -> None:
    """Exercise the public train -> artifact -> inference path on repository data."""
    X, y = load_labeled_data(Path("clean_data"), 15)

    assert not X.empty
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y)

    sample_index = list(y[y == 0].head(25).index) + list(y[y == 1].head(25).index)
    X_sample = X.loc[sample_index].copy()
    y_sample = y.loc[sample_index].copy()

    pipeline = build_pipeline(X_sample, random_state=42)
    pipeline.set_params(
        model__n_estimators=5,
        model__max_depth=2,
        model__n_jobs=1,
    )
    pipeline.fit(X_sample, y_sample)

    metadata = {
        "window_seconds": 15,
        "feature_columns": list(X.columns),
    }
    artifact_path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, artifact_path)

    loaded_pipeline, loaded_metadata = load_bundle(artifact_path)
    result = score_frame(X_sample.head(8), loaded_pipeline, loaded_metadata)

    assert len(result) == 8
    assert set(result["prediction"].unique()).issubset({0, 1})
    assert result["vpn_probability"].between(0.0, 1.0).all()


def test_prepare_features_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"duration": [1.0, 2.0]})
    metadata = {"feature_columns": ["duration", "flowBytesPerSecond"]}

    with pytest.raises(ValueError, match="missing 1 required column"):
        prepare_features(frame, metadata)
