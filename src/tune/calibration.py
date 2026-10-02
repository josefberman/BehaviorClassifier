"""Compare sim feature statistics against a calibration target report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from scripts.generate_sims import _event_window, _overrides_for
from src.features.order_params import FEATURE_NAMES
from src.features.windows import frame_feature_matrix
from src.labels import CANONICAL
from src.sim.config import deep_merge
from src.sim.model_fast import run_simulation_fast

FEATURE_WEIGHTS: dict[str, float] = {name: 1.0 for name in FEATURE_NAMES}
FEATURE_STD_FLOOR: dict[str, float] = {name: 0.03 for name in FEATURE_NAMES}


def _n_samples(block: dict) -> int:
    return int(block.get("n_frames", block.get("n_segments", block.get("n_clips", 0))))


def load_calibration_targets(path: Path) -> dict[str, dict]:
    """Load per-behavior feature stats from calibration_report.json."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if "real" in payload:
        return payload["real"]
    if "sim" in payload and len(payload) == 1:
        return payload["sim"]
    return payload


def _aggregate(rows: dict[str, list[np.ndarray]]) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for behavior in sorted(rows):
        xs = np.vstack(rows[behavior])
        feat_stats = {}
        for i, name in enumerate(FEATURE_NAMES):
            col = xs[:, i]
            feat_stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
            }
        report[behavior] = {
            "n_frames": int(len(xs)),
            "features": feat_stats,
        }
    return report


def _run_one(
    behavior: str,
    n: int,
    seed: int,
    overrides: dict,
) -> np.ndarray | None:
    ov = _overrides_for(behavior, seed)
    if overrides:
        ov = deep_merge(ov, overrides)
    use_seed = seed + n * 10007
    try:
        result = run_simulation_fast(behavior, n, use_seed, overrides=ov)
    except Exception:
        return None
    pos, vel = result.positions, result.velocities
    es, ee = _event_window(behavior, pos.shape[0])
    if es is not None:
        pos, vel = pos[es:ee], vel[es:ee]
    if pos.shape[0] < 15:
        return None
    return frame_feature_matrix(pos, vel, fps=30.0)


def summarize_behavior_sims(
    behavior: str,
    overrides: dict,
    *,
    n_values: list[int],
    n_seeds: int,
    n_jobs: int = -1,
) -> dict[str, dict]:
    """Run sims for one behavior and return aggregated feature stats."""
    tasks = [(n, seed) for n in n_values for seed in range(n_seeds)]
    rows_list = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(behavior, n, seed, overrides) for n, seed in tasks
    )
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for feat in rows_list:
        if feat is not None:
            rows[behavior].append(feat)
    if not rows:
        empty = {name: {"mean": 0.0, "std": 0.0} for name in FEATURE_NAMES}
        return {behavior: {"n_frames": 0, "features": empty}}
    return _aggregate(rows)


def behavior_calibration_loss(
    sim_block: dict,
    target_block: dict,
) -> float:
    """Weighted MSE of sim vs real feature means and stds (lower is better).

    Each Φ contributes equally-weighted mean and std residuals, scaled by the
    real-frame std so a 0.05 miss costs the same for location and spread.
    """
    if _n_samples(sim_block) == 0:
        return float("inf")

    total = 0.0
    n = 0
    for feat in FEATURE_NAMES:
        stats_s = sim_block["features"][feat]
        stats_t = target_block["features"][feat]
        sm = float(stats_s["mean"])
        ss = float(stats_s["std"])
        tm = float(stats_t["mean"])
        ts = float(stats_t["std"])
        scale = max(ts, FEATURE_STD_FLOOR[feat])
        w = FEATURE_WEIGHTS[feat]
        total += w * (((sm - tm) / scale) ** 2 + ((ss - ts) / scale) ** 2)
        n += 1
    return total / max(n, 1)


def total_calibration_loss(
    sim_report: dict[str, dict],
    target_report: dict[str, dict],
    *,
    behaviors: list[str] | None = None,
) -> float:
    behaviors = behaviors or list(CANONICAL)
    losses = []
    for behavior in behaviors:
        if behavior not in target_report:
            continue
        sim_block = sim_report.get(
            behavior,
            {"n_frames": 0, "features": {}},
        )
        losses.append(behavior_calibration_loss(sim_block, target_report[behavior]))
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def score_from_loss(loss: float) -> float:
    """Higher-is-better score for logging (negative loss)."""
    if loss == float("inf"):
        return -1e9
    return -loss
