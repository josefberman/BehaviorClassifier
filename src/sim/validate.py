"""Soft validation of simulated behaviours via order-parameter signatures."""

from __future__ import annotations

import numpy as np

from src.features.order_params import compute_order_params_series

# Qualitative bands on Φ_trans and anisotropy-corrected Φ_tan / Φ_rad^±.
# Numerical tan/rad thresholds from the old uncorrected features do not apply.
_BOUNDS: dict[str, dict[str, float]] = {
    "traveling_polarized": {
        "phi_trans_min": 0.65,
        "phi_tan_max": 0.55,
    },
    "milling": {
        "phi_tan_min": 0.35,
        "phi_trans_max": 0.61,
    },
    "shoaling": {
        "phi_trans_max": 0.72,
        "phi_rad_abs_max": 0.40,
    },
    "expansion_burst": {
        "phi_rad_min": 0.0,
    },
    "compaction": {
        "phi_rad_max": 0.0,
    },
}


def summarize_metrics(positions: np.ndarray, velocities: np.ndarray) -> dict[str, float]:
    series = compute_order_params_series(positions, velocities)
    out = {}
    for k, arr in series.items():
        out[f"{k}_mean"] = float(np.mean(arr))
        out[f"{k}_std"] = float(np.std(arr))
    out["phi_rad_event"] = float(np.mean(series["phi_rad_pm"]))
    return out


def metrics_for_validation(
    behavior: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    event_start: int | None = None,
    event_end: int | None = None,
) -> dict[str, float]:
    """Compute validation metrics on phi_trans, phi_tan, and phi_rad_pm only."""
    pos, vel = positions, velocities
    if event_start is not None and event_end is not None:
        a = max(0, int(event_start))
        b = min(int(event_end), pos.shape[0])
        if b > a:
            pos, vel = pos[a:b], vel[a:b]
    return summarize_metrics(pos, vel)


def validate_behavior(
    behavior: str,
    metrics: dict[str, float],
    positions: np.ndarray | None = None,
    velocities: np.ndarray | None = None,
) -> bool:
    phi_trans = metrics["phi_trans_mean"]
    phi_tan = metrics["phi_tan_mean"]
    phi_rad = metrics["phi_rad_pm_mean"]

    if behavior == "traveling_polarized":
        b = _BOUNDS[behavior]
        return phi_trans > b["phi_trans_min"] and phi_tan < b["phi_tan_max"]
    if behavior == "milling":
        b = _BOUNDS[behavior]
        return phi_tan > b["phi_tan_min"] and phi_trans < b["phi_trans_max"]
    if behavior == "shoaling":
        b = _BOUNDS[behavior]
        return phi_trans < b["phi_trans_max"] and abs(phi_rad) < b["phi_rad_abs_max"]
    if behavior == "expansion_burst":
        b = _BOUNDS[behavior]
        return phi_rad > b["phi_rad_min"]
    if behavior == "compaction":
        b = _BOUNDS[behavior]
        return phi_rad < b["phi_rad_max"]
    return True
