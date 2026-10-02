"""Random search to match sim order-parameter stats to real calibration targets."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.order_params import FEATURE_NAMES
from src.labels import CANONICAL, canonicalize
from src.tune.calibration import (
    behavior_calibration_loss,
    load_calibration_targets,
    score_from_loss,
    summarize_behavior_sims,
    total_calibration_loss,
)
from src.tune.search_space import random_samples, round_overrides


def _feat_progress(sf: dict) -> str:
    parts = []
    for name in FEATURE_NAMES:
        short = name.removeprefix("phi_")
        m = sf[name]["mean"]
        s = sf[name]["std"]
        if name == "phi_rad_pm":
            parts.append(f"{short}={m:+.3f}±{s:.3f}")
        else:
            parts.append(f"{short}={m:.3f}±{s:.3f}")
    return "  ".join(parts)


def _canonical_behavior_map(d: dict[str, Any] | None) -> dict[str, Any]:
    """Remap keys such as swarming → shoaling; drop unknown labels."""
    out: dict[str, Any] = {}
    for key, val in (d or {}).items():
        try:
            canon = canonicalize(str(key))
        except ValueError:
            continue
        if canon not in CANONICAL:
            continue
        out[canon] = val
    return out


TUNING_DIR = ROOT / "results" / "tuning" / "feature_match"
BEST_PATH = TUNING_DIR / "best.json"
HISTORY_PATH = TUNING_DIR / "history.jsonl"


def tune_behavior(
    behavior: str,
    target_report: dict[str, dict],
    *,
    n_trials: int = 500,
    n_seeds: int = 16,
    n_values: list[int] | None = None,
    n_jobs: int = -1,
    rng: np.random.Generator | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Random search of overrides for a single behavior against real feature targets."""
    if behavior not in target_report:
        raise ValueError(f"No target stats for {behavior}")

    n_values = n_values or [20, 40]
    rng = rng or np.random.default_rng()
    target_block = target_report[behavior]
    candidates = random_samples(behavior, n_trials, rng)
    if show_progress:
        print(f"  random search: {len(candidates)} points in {len(candidates[0]) if candidates else 0}D")
        print(f"  matching mean+std of {len(FEATURE_NAMES)} frame features: {', '.join(FEATURE_NAMES)}")
        print(f"  target  {_feat_progress(target_block['features'])}")

    best_loss = float("inf")
    best_overrides: dict = {}
    trials: list[dict[str, Any]] = []

    for trial_idx, ov in enumerate(candidates):
        sim_report = summarize_behavior_sims(
            behavior,
            ov,
            n_values=n_values,
            n_seeds=n_seeds,
            n_jobs=n_jobs,
        )
        loss = behavior_calibration_loss(sim_report[behavior], target_block)
        score = score_from_loss(loss)
        record = {
            "behavior": behavior,
            "trial": trial_idx + 1,
            "loss": loss,
            "score": score,
            "n_frames": sim_report[behavior]["n_frames"],
            "overrides": round_overrides({behavior: ov})[behavior],
            "sim_features": sim_report[behavior]["features"],
        }
        trials.append(record)

        if show_progress:
            print(
                f"  trial {trial_idx + 1:4d}/{n_trials}  loss={loss:.4f}  "
                f"{_feat_progress(sim_report[behavior]['features'])}"
            )

        if loss < best_loss:
            best_loss = loss
            best_overrides = ov
            if show_progress:
                print(f"    new best (loss={best_loss:.4f})")

    return {
        "behavior": behavior,
        "best_loss": best_loss,
        "best_score": score_from_loss(best_loss),
        "best_overrides": round_overrides({behavior: best_overrides})[behavior],
        "target_features": target_block["features"],
        "trials": trials,
    }


def tune_all_behaviors(
    target_path: Path,
    *,
    behaviors: list[str] | None = None,
    n_trials: int = 500,
    n_seeds: int = 16,
    n_values: list[int] | None = None,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Tune each behavior independently, then merge best overrides."""
    target_report = load_calibration_targets(target_path)
    behaviors = behaviors or [b for b in CANONICAL if b in target_report]
    rng = np.random.default_rng()

    # Merge with any existing best overrides for behaviors we skip.
    merged_overrides: dict[str, dict] = {}
    if BEST_PATH.exists():
        with open(BEST_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        merged_overrides.update(_canonical_behavior_map(prev.get("behavior_overrides")))

    per_behavior: dict[str, Any] = {}

    print(f"Target report: {target_path}")
    print(f"Behaviors: {behaviors}\n")

    for behavior in behaviors:
        print(f"=== {behavior} ===")
        result = tune_behavior(
            behavior,
            target_report,
            n_trials=n_trials,
            n_seeds=n_seeds,
            n_values=n_values,
            n_jobs=n_jobs,
            rng=rng,
            show_progress=show_progress,
        )
        per_behavior[behavior] = result
        merged_overrides[behavior] = result["best_overrides"]
        append_history(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "behavior": behavior,
                "best_loss": result["best_loss"],
                "best_overrides": result["best_overrides"],
            }
        )
        print()

    sim_report: dict[str, dict] = {}
    for behavior, ov in merged_overrides.items():
        if behavior not in target_report:
            continue
        sim_report.update(
            summarize_behavior_sims(
                behavior,
                ov,
                n_values=n_values or [20, 40],
                n_seeds=n_seeds,
                n_jobs=n_jobs,
            )
        )

    prev_per_behavior: dict[str, Any] = {}
    if BEST_PATH.exists():
        with open(BEST_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        prev_per_behavior = _canonical_behavior_map(prev.get("per_behavior"))

    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "target_path": str(target_path),
        "n_trials_per_behavior": n_trials,
        "n_seeds": n_seeds,
        "n_values": n_values or [20, 40],
        "total_loss": total_calibration_loss(
            sim_report,
            target_report,
            behaviors=[b for b in CANONICAL if b in target_report],
        ),
        "behavior_overrides": round_overrides(merged_overrides),
        "per_behavior": {
            **prev_per_behavior,
            **{
                b: {
                    "best_loss": r["best_loss"],
                    "best_overrides": r["best_overrides"],
                }
                for b, r in per_behavior.items()
            },
        },
        "sim_features": {b: sim_report[b]["features"] for b in sim_report},
        "target_features": {b: target_report[b]["features"] for b in target_report if b in CANONICAL},
    }
    save_best(summary)
    return summary


def append_history(record: dict[str, Any]) -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def save_best(record: dict[str, Any]) -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
