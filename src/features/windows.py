"""Non-overlapping 0.5 s windows of order-parameter means."""

from __future__ import annotations

import numpy as np

from src.features.order_params import FEATURE_NAMES, compute_order_params_series

WINDOW_SEC = 0.5


def window_length(fps: float, window_sec: float = WINDOW_SEC) -> int:
    return max(1, int(round(window_sec * fps)))


def frame_feature_matrix(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    names: list[str] | None = None,
    fps: float = 30.0,
) -> np.ndarray:
    """Per-frame order parameters (T, 5)."""
    del fps
    names = names or list(FEATURE_NAMES)
    series = compute_order_params_series(positions, velocities)
    t = next(iter(series.values())).shape[0] if series else positions.shape[0]
    return np.column_stack(
        [np.asarray(series.get(name, np.zeros(t)), dtype=np.float64) for name in names]
    )


def window_feature_matrix(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    window_sec: float = WINDOW_SEC,
    hop_sec: float | None = None,
    fps: float = 30.0,
    names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Means of the five order parameters over non-overlapping windows.

    Windows shorter than `window_sec` are dropped. Returns (n_windows, 5)
    features and the start frame of each window (relative to `positions`).
    """
    hop_sec = window_sec if hop_sec is None else hop_sec
    names = names or list(FEATURE_NAMES)
    X = frame_feature_matrix(positions, velocities, names=names, fps=fps)
    t = X.shape[0]
    w = window_length(fps, window_sec)
    h = max(1, int(round(hop_sec * fps)))
    if t < w:
        return np.zeros((0, len(names)), dtype=np.float64), np.zeros(0, dtype=np.int64)
    starts = np.arange(0, t - w + 1, h, dtype=np.int64)
    rows = np.stack([X[s : s + w].mean(axis=0) for s in starts])
    return rows, starts
