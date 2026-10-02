#!/usr/bin/env python3
"""Predict a behavior label for every frame of a trajectory CSV/H5."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classify.predict import predict_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a fish-trajectory CSV/H5, predict a behavior per frame with a "
            "trained classifier, and write the same file type plus a label column."
        )
    )
    parser.add_argument("trajectory", type=Path, help="Input .csv / .h5 / .hdf5 trajectory")
    parser.add_argument("model", type=Path, help="Trained classifier.joblib")
    parser.add_argument("output", type=Path, help="Output path (same suffix as input, or omit suffix)")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30)")
    parser.add_argument(
        "--column",
        default="predicted_behavior",
        help="Name of the written label column (default: predicted_behavior)",
    )
    args = parser.parse_args()

    df = predict_trajectory(
        args.trajectory,
        args.model,
        args.output,
        fps=args.fps,
        column=args.column,
    )
    counts = Counter(df[args.column].tolist())
    print(f"wrote {args.output}  frames={len(df)}")
    for label, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
