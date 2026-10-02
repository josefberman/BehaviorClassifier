"""Per-window behavior predictions from a trained classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.order_params import FEATURE_NAMES
from src.features.windows import WINDOW_SEC, window_feature_matrix, window_length
from src.io import load_trajectory_table, pos_vel_from_table, save_trajectory_table

ROOT = Path(__file__).resolve().parents[2]


def predict_trajectory(
    trajectory_path: Path,
    model_path: Path | None = None,
    out_path: Path | None = None,
    *,
    fps: float = 30.0,
    column: str = "predicted_behavior",
) -> pd.DataFrame:
    """Label each frame from 0.5 s windows and write a copy with a behavior column."""
    trajectory_path = Path(trajectory_path)
    model_path = Path(model_path or (ROOT / "results" / "classifier.joblib"))
    out_path = Path(out_path) if out_path is not None else None

    df = load_trajectory_table(trajectory_path)
    pos, vel = pos_vel_from_table(df, fps=fps)
    if pos.shape[0] == 0:
        raise ValueError(f"No frames in {trajectory_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]
    names = list(bundle.get("feature_names") or FEATURE_NAMES)
    window_sec = float(bundle.get("window_sec", WINDOW_SEC))

    X, starts = window_feature_matrix(pos, vel, window_sec=window_sec, fps=fps, names=names)
    t = pos.shape[0]
    labels = np.empty(t, dtype=object)
    labels[:] = ""
    if len(X) == 0:
        raise ValueError(f"Trajectory shorter than one {window_sec}s window")
    pred = le.inverse_transform(np.asarray(model.predict(X)))
    w = window_length(fps, window_sec)
    for start, lab in zip(starts, pred):
        labels[int(start) : int(start) + w] = lab
    last_end = int(starts[-1]) + w
    if last_end < t:
        labels[last_end:] = pred[-1]

    out = df.copy()
    out[column] = labels
    if out_path is not None:
        if out_path.suffix == "":
            out_path = out_path.with_suffix(trajectory_path.suffix)
        save_trajectory_table(out_path, out)
    return out
