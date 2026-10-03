#!/usr/bin/env python3
"""Train XGBoost on 0.5 s windows from annotated real trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify.train import train_classifier
from src.features.windows import WINDOW_SEC


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a leave-one-video-out XGBoost classifier on annotated schooling videos."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for classifier.joblib and cv_metrics.json",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=WINDOW_SEC,
        help="Window length in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--no-transitions",
        "--stable-only",
        action="store_true",
        help="Train and evaluate on the five behaviors only; drop transition windows",
    )
    args = parser.parse_args()

    report = train_classifier(
        out_dir=args.out_dir,
        window_sec=args.window_sec,
        include_transitions=not args.no_transitions,
    )
    oof = report["oof"]
    summary = {
        "n_windows": report["n_windows"],
        "n_videos": report["n_videos"],
        "include_transitions": report["include_transitions"],
        "params": report["params"],
        "oof_macro_f1": oof["macro_f1"],
        "oof_balanced_accuracy": oof["balanced_accuracy"],
        "oof_accuracy": oof["accuracy"],
    }
    print(json.dumps(summary, indent=2))
    print(f"oof_macro_f1={oof['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
