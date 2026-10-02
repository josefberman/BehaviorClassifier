"""End-to-end trial: generate sims → train → real eval."""

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

from scripts.generate_sims import generate_batch
from src.classify.eval import eval_real
from src.classify.train import train_classifier
from src.tune.search_space import round_overrides, sample_overrides

TUNING_DIR = ROOT / "results" / "tuning"
HISTORY_PATH = TUNING_DIR / "history.jsonl"
BEST_PATH = TUNING_DIR / "best.json"


def score_metrics(metrics: dict[str, Any], *, objective: str = "real_macro_f1") -> float:
    if metrics.get("skipped"):
        return -1.0

    def _get(key: str) -> float:
        val = metrics.get(key)
        return -1.0 if val is None else float(val)

    if objective == "real_accuracy":
        return _get("real_accuracy")
    if objective == "combined":
        f1 = _get("real_macro_f1")
        acc = _get("real_accuracy")
        if f1 < 0 or acc < 0:
            return -1.0
        return 0.7 * f1 + 0.3 * acc
    return _get("real_macro_f1")


def run_trial(
    trial_dir: Path,
    behavior_overrides: dict[str, dict],
    *,
    n_seeds: int = 40,
    n_values: list[int] | None = None,
    n_jobs: int = -1,
    mode: str = "frame",
    min_valid_frac: float = 0.85,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Generate sims, train classifier, evaluate on real annotations."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    sim_root = trial_dir / "sim_datasets"
    n_values = n_values or [10, 30, 50]

    with open(trial_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(round_overrides(behavior_overrides), f, indent=2)
        f.write("\n")

    results = generate_batch(
        sim_root,
        n_seeds=n_seeds,
        n_values=n_values,
        n_jobs=n_jobs,
        behavior_overrides=behavior_overrides,
        show_progress=show_progress,
    )
    n_ok = sum(1 for r in results if r.get("valid"))
    valid_frac = n_ok / max(len(results), 1)

    metrics: dict[str, Any] = {
        "trial_dir": str(trial_dir),
        "n_sims": len(results),
        "n_valid": n_ok,
        "valid_frac": valid_frac,
        "behavior_overrides": round_overrides(behavior_overrides),
    }

    if valid_frac < min_valid_frac:
        metrics["skipped"] = True
        metrics["reason"] = f"valid_frac {valid_frac:.3f} < {min_valid_frac}"
        metrics["real_macro_f1"] = None
        metrics["real_accuracy"] = None
        with open(trial_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")
        return metrics

    manifest_path = sim_root / "manifest.json"
    train_report = train_classifier(
        out_dir=trial_dir,
        mode=mode,
        manifest_path=manifest_path,
        sim_root=sim_root,
    )
    real_report = eval_real(model_path=trial_dir / "classifier.joblib", out_dir=trial_dir)

    metrics.update(
        {
            "sim_test_macro_f1": train_report.get("sim_test_macro_f1"),
            "sim_test_accuracy": train_report.get("sim_test_accuracy"),
            "real_macro_f1": real_report.get("macro_f1"),
            "real_accuracy": real_report.get("accuracy"),
            "n_train": train_report.get("n_train"),
            "n_test": train_report.get("n_test"),
            "n_real": real_report.get("n_real"),
        }
    )
    with open(trial_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    return metrics


def append_history(record: dict[str, Any]) -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_best() -> dict[str, Any] | None:
    if not BEST_PATH.exists():
        return None
    with open(BEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_best(record: dict[str, Any]) -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")


def tune_loop(
    *,
    n_trials: int = 10,
    n_seeds: int = 40,
    n_values: list[int] | None = None,
    n_jobs: int = -1,
    objective: str = "real_macro_f1",
    jitter: float = 0.35,
    mode: str = "frame",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Random-search loop optimizing real eval metrics."""
    rng = np.random.default_rng()
    best = load_best()
    best_score = score_metrics(best, objective=objective) if best else -1.0
    center = best.get("behavior_overrides") if best else None

    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "n_trials": n_trials,
        "objective": objective,
        "trials": [],
        "best_score_before": best_score,
    }

    for trial_idx in range(n_trials):
        trial_id = f"trial_{trial_idx + 1:04d}"
        trial_dir = TUNING_DIR / trial_id
        use_jitter = jitter if center else 0.0
        overrides = sample_overrides(rng, center=center, jitter=use_jitter)

        print(f"\n=== {trial_id} ===")
        metrics = run_trial(
            trial_dir,
            overrides,
            n_seeds=n_seeds,
            n_values=n_values,
            n_jobs=n_jobs,
            mode=mode,
            show_progress=show_progress,
        )
        trial_score = score_metrics(metrics, objective=objective)
        record = {
            "trial_id": trial_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score": trial_score,
            "objective": objective,
            **metrics,
        }
        append_history(record)
        summary["trials"].append(
            {
                "trial_id": trial_id,
                "score": trial_score,
                "real_macro_f1": metrics.get("real_macro_f1"),
                "real_accuracy": metrics.get("real_accuracy"),
                "sim_test_macro_f1": metrics.get("sim_test_macro_f1"),
                "valid_frac": metrics.get("valid_frac"),
            }
        )

        print(
            f"score={trial_score:.4f}  real_f1={metrics.get('real_macro_f1')}  "
            f"real_acc={metrics.get('real_accuracy')}  sim_f1={metrics.get('sim_test_macro_f1')}"
        )

        if trial_score > best_score:
            best_score = trial_score
            center = overrides
            save_best(record)
            print(f"  new best ({objective}={best_score:.4f})")

    summary["best_score_after"] = best_score
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    with open(TUNING_DIR / "last_run.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary
