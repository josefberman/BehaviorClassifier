#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify.train import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classifier on simulated features.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--transitions",
        action="store_true",
        default=None,
        help="Include X_to_Y transition clips from the sim manifest (default: auto-detect)",
    )
    group.add_argument(
        "--no-transitions",
        "--stable-only",
        action="store_false",
        dest="transitions",
        help="Train on baseline clips only (tpol, milling, shoaling, e+, e−); exclude transitions",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "sim_datasets" / "manifest.json",
        help="Path to sim manifest.json",
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=None,
        help="Sim dataset root (default: parent of manifest)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for classifier.joblib and sim_test_metrics.json",
    )
    args = parser.parse_args()

    sim_root = args.sim_root or args.manifest.parent
    stable_only = args.transitions is False
    report = train_classifier(
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        sim_root=sim_root,
        include_transitions=False if stable_only else args.transitions,
        stable_only=stable_only,
    )
    summary = {k: report[k] for k in report if k != "classification_report"}
    print(json.dumps(summary, indent=2))
    print(f"sim_macro_f1={report['best_sim_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
