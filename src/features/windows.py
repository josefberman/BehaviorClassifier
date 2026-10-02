"""Sliding-window aggregation and per-frame stacking of order parameters."""

from __future__ import annotations

import numpy as np

from src.features.order_params import (
    AGG_FEATURE_NAMES,
    FEATURE_NAMES,
    aggregate_series,
    compute_order_params_series,
)


def segment_feature_vector(
    positions: np.ndarray,
    velocities: np.ndarray,
    fps: float = 30.0,
) -> dict[str, float]:
    series = compute_order_params_series(positions, velocities)
    return aggregate_series(series, fps=fps)


def sliding_window_features(
    positions: np.ndarray,
    velocities: np.ndarray,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    fps: float = 30.0,
) -> list[dict[str, float]]:
    w = max(2, int(round(window_sec * fps)))
    h = max(1, int(round(hop_sec * fps)))
    t = positions.shape[0]
    feats = []
    for start in range(0, max(1, t - w + 1), h):
        end = min(t, start + w)
        if end - start < max(2, w // 2):
            continue
        feats.append(
            segment_feature_vector(positions[start:end], velocities[start:end], fps=fps)
        )
    if not feats:
        feats.append(segment_feature_vector(positions, velocities, fps=fps))
    return feats


def feature_dict_to_array(feat: dict[str, float], names: list[str] | None = None) -> np.ndarray:
    names = names or AGG_FEATURE_NAMES
    return np.array([feat.get(n, 0.0) for n in names], dtype=np.float64)


def frame_feature_matrix(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    names: list[str] | None = None,
    fps: float = 30.0,
) -> np.ndarray:
    """Per-frame classifier inputs: instantaneous order parameters."""
    del fps
    names = names or list(FEATURE_NAMES)
    series = compute_order_params_series(positions, velocities)
    t = next(iter(series.values())).shape[0] if series else positions.shape[0]
    return np.column_stack([np.asarray(series.get(name, np.zeros(t)), dtype=np.float64) for name in names])
