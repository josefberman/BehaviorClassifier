#!/usr/bin/env python3
"""Summarize order-parameter features over generated sims and/or real annotations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.dataset import (
    features_from_sim_entry,
    load_manifest,
    load_real_segments,
)
from src.features.order_params import FEATURE_NAMES
from src.features.windows import frame_feature_matrix
from src.labels import CANONICAL, is_transition
from src.sim.io import load_trajectory_csv


def _aggregate(
    rows: dict[str, list[np.ndarray]],
    *,
    n_segments: dict[str, int] | None = None,
) -> dict[str, dict]:
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
            "n_segments": int((n_segments or {}).get(behavior, 0)),
            "features": feat_stats,
        }
    return report


def _summarize_sim(
    manifest_path: Path,
    sim_root: Path,
    *,
    include_transitions: bool = False,
    valid_only: bool = True,
) -> dict[str, dict]:
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    n_segs: dict[str, int] = defaultdict(int)
    allowed = set(CANONICAL)
    for entry in load_manifest(manifest_path):
        behavior = entry.get("behavior", "")
        if not include_transitions:
            if is_transition(behavior) or behavior not in allowed:
                continue
        if valid_only and not entry.get("valid", True):
            continue
        csv_path = sim_root / entry["csv"]
        if not csv_path.exists():
            continue
        added = None
        for feat_vec, label in features_from_sim_entry(entry, mode="frame", sim_root=sim_root):
            key = label if include_transitions else behavior
            rows[key].append(feat_vec)
            added = key
        if added is not None:
            n_segs[added] += 1
    return _aggregate(rows, n_segments=n_segs)


def _summarize_real(
    *,
    annotations_dir: Path | None = None,
    include_transitions: bool = False,
    min_frames: int = 15,
) -> dict[str, dict]:
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    n_segs: dict[str, int] = defaultdict(int)
    allowed = set(CANONICAL)
    for seg in load_real_segments(annotations_dir=annotations_dir):
        label = seg["label"]
        if not include_transitions:
            if is_transition(label) or label not in allowed:
                continue
        pos, vel = load_trajectory_csv(ROOT / seg["csv"])
        a, b = seg["start"], min(seg["end"], pos.shape[0])
        if b - a < min_frames:
            continue
        X = frame_feature_matrix(pos[a:b], vel[a:b], fps=seg["fps"])
        if X.shape[0] == 0:
            continue
        rows[label].append(X)
        n_segs[label] += 1
    return _aggregate(rows, n_segments=n_segs)


def _print_report(report: dict[str, dict], *, title: str) -> None:
    print(title)
    for behavior, block in report.items():
        n = block.get("n_frames", block.get("n_segments", block.get("n_clips", 0)))
        n_seg = block.get("n_segments")
        extra = f", {n_seg} segments" if n_seg else ""
        print(f"{behavior}  (n={n} frames{extra})")
        for name in FEATURE_NAMES:
            stats = block["features"][name]
            print(f"  {name:16s}  mean={stats['mean']:+.4f}  std={stats['std']:.4f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mean/std of per-frame order parameters per behavior (sim manifest and/or real annotations).",
    )
    parser.add_argument(
        "--source",
        choices=("sim", "real", "both"),
        default="sim",
        help="Data source (default: sim)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "sim_datasets" / "manifest.json",
        help="Path to sim manifest.json from generate_sims.py",
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=None,
        help="Sim dataset root (default: parent of manifest)",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=ROOT / "annotations",
        help="Directory containing *_motion.json annotation files",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=15,
        help="Minimum segment length for real annotations (default: 15 frames)",
    )
    parser.add_argument(
        "--transitions",
        action="store_true",
        help="Include transition-labelled segments/clips",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include sim manifest entries marked valid=false",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "calibration_report.json",
        help="JSON output path",
    )
    args = parser.parse_args()

    payload: dict = {}
    printed = False

    if args.source in ("sim", "both"):
        sim_root = args.sim_root or args.manifest.parent
        if not args.manifest.exists():
            raise SystemExit(f"Manifest not found: {args.manifest}  (run generate_sims.py first)")
        sim_report = _summarize_sim(
            args.manifest,
            sim_root,
            include_transitions=args.transitions,
            valid_only=not args.include_invalid,
        )
        if not sim_report:
            raise SystemExit(f"No sim clips found under {sim_root}")
        _print_report(sim_report, title="=== Simulations (generate_sims.py) ===")
        printed = True
        payload["sim"] = sim_report

    if args.source in ("real", "both"):
        real_report = _summarize_real(
            annotations_dir=args.annotations_dir,
            include_transitions=args.transitions,
            min_frames=args.min_frames,
        )
        if not real_report:
            raise SystemExit(
                f"No real segments found in {args.annotations_dir} "
                "(check annotations and schooling-datasets paths)"
            )
        if printed:
            print()
        _print_report(real_report, title="=== Real annotations ===")
        payload["real"] = real_report if args.source == "both" else real_report

    if args.source != "both":
        payload = payload.get(args.source, payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
