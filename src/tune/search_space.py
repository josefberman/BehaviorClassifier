"""Search space for behavior YAML overrides."""

from __future__ import annotations

import copy
from typing import Any

from src.labels import CANONICAL

# Ranges are (low, high) inclusive for uniform sampling.
# Interaction radii r_r=30, r_o=90, r_a=150 are fixed and not tuned.
# Bounds cover the model defaults and current YAMLs without allowing
# one-frame π turns or unbounded social torques.
_PARAM_SPACE: dict[str, tuple[float, float]] = {
    "w_r": (0.5, 4.0),
    "w_o": (0.0, 2.5),
    "w_a": (0.0, 2.5),
    "w_tan": (0.0, 2.0),
    "w_rad": (-2.0, 2.0),
    "sigma_theta": (0.0, 0.25),
    "s_0": (0.3, 2.0),
    "sigma_s": (0.0, 0.20),
    "omega_max": (0.02, 0.30),
    "a_max": (0.05, 1.50),
}
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    behavior: dict(_PARAM_SPACE) for behavior in CANONICAL
}

_INT_KEYS: set[str] = set()


def _sample_leaf(rng, spec: tuple[float, float], *, jitter: float, center_val: float | None):
    lo, hi = spec
    if center_val is not None and jitter > 0:
        c = min(max(float(center_val), lo), hi)
        span = (hi - lo) * jitter
        lo = max(lo, c - span)
        hi = min(hi, c + span)
    if hi <= lo:
        if center_val is not None:
            return float(min(max(float(center_val), spec[0]), spec[1]))
        return float((spec[0] + spec[1]) / 2)
    return float(rng.uniform(lo, hi))


def _sample_node(rng, space: dict, center: dict | None, jitter: float) -> dict:
    out: dict = {}
    for key, spec in space.items():
        csub = (center or {}).get(key)
        if isinstance(spec, dict):
            out[key] = _sample_node(rng, spec, csub if isinstance(csub, dict) else None, jitter)
        elif isinstance(spec, tuple) and len(spec) == 2:
            val = _sample_leaf(rng, spec, jitter=jitter, center_val=csub if isinstance(csub, (int, float)) else None)
            if key in _INT_KEYS:
                val = int(round(val))
            out[key] = val
    return out


def sample_overrides(
    rng,
    *,
    behaviors: list[str] | None = None,
    center: dict[str, dict] | None = None,
    jitter: float = 0.0,
) -> dict[str, dict]:
    """Sample a full behavior-overrides dict from SEARCH_SPACE."""
    behaviors = behaviors or list(CANONICAL)
    out: dict[str, dict] = {}
    for behavior in behaviors:
        if behavior not in SEARCH_SPACE:
            continue
        out[behavior] = _sample_node(rng, SEARCH_SPACE[behavior], (center or {}).get(behavior), jitter)
    return out


def random_samples(
    behavior: str,
    n: int,
    rng,
) -> list[dict[str, float]]:
    """n independent uniform draws from SEARCH_SPACE[behavior]."""
    if n <= 0 or behavior not in SEARCH_SPACE:
        return []
    return [_sample_node(rng, SEARCH_SPACE[behavior], None, 0.0) for _ in range(n)]


def round_overrides(overrides: dict[str, dict]) -> dict[str, dict]:
    """Deep-copy overrides with stable float rounding for JSON logs."""

    def _round(obj):
        if isinstance(obj, dict):
            return {k: _round(v) for k, v in obj.items()}
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    return _round(copy.deepcopy(overrides))
