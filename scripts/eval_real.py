#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify.eval import eval_real


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classifier on real annotated segments.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--transitions",
        action="store_true",
        default=None,
        help="Include transition-labelled real segments (default: auto-detect from model)",
    )
    group.add_argument(
        "--no-transitions",
        "--stable-only",
        action="store_false",
        dest="transitions",
        help="Evaluate baseline real segments only (incl. e+/e−); exclude transition intervals",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "results" / "classifier.joblib",
        help="Path to trained classifier.joblib",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for real_eval_metrics.json",
    )
    args = parser.parse_args()

    stable_only = args.transitions is False
    report = eval_real(
        model_path=args.model,
        out_dir=args.out_dir,
        include_transitions=False if stable_only else args.transitions,
        stable_only=stable_only,
    )
    summary = {k: report[k] for k in report if k != "classification_report"}
    print(json.dumps(summary, indent=2))
    if "accuracy" in report:
        print(
            f"real_balanced_accuracy={report.get('balanced_accuracy', float('nan')):.3f} "
            f"macro_f1={report['macro_f1']:.3f}  accuracy={report['accuracy']:.3f}"
        )
        if report.get("include_transitions") and "transition_macro_f1" in report:
            print(f"real_baseline_macro_f1={report.get('baseline_macro_f1', float('nan')):.3f}")
            print(f"real_transition_macro_f1={report['transition_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
