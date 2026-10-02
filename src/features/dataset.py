"""Build 0.5 s window feature matrices from annotated real trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.features.order_params import FEATURE_NAMES
from src.features.windows import WINDOW_SEC, window_feature_matrix
from src.io import load_motion_json, load_trajectory_csv, mmss_to_frame
from src.labels import canonicalize, load_aliases

ROOT = Path(__file__).resolve().parents[2]


def load_real_segments(
    annotations_dir: Path | None = None,
    datasets_json: Path | None = None,
) -> list[dict]:
    annotations_dir = annotations_dir or (ROOT / "annotations")
    datasets_json = datasets_json or (annotations_dir / "datasets.json")
    with open(datasets_json, encoding="utf-8") as f:
        meta = json.load(f)
    aliases = load_aliases()
    out = []
    for path in sorted(annotations_dir.glob("*_motion.json")):
        if path.name.startswith("_"):
            continue
        ann = load_motion_json(path)
        ds = ann["dataset"]
        if ds not in meta:
            continue
        fps = float(ann.get("fps", meta[ds]["fps"]))
        group = meta[ds]["fish_group"]
        csv = ROOT / "schooling-datasets" / group / ds / f"{ds}_loc_vel_data.csv"
        if not csv.exists():
            continue
        min_len = max(1, int(round(WINDOW_SEC * fps)))
        for seg in ann["segments"]:
            try:
                label = canonicalize(seg["label"], aliases)
            except ValueError:
                continue
            start = mmss_to_frame(seg["start"], fps)
            end = mmss_to_frame(seg["end"], fps)
            if end - start < min_len:
                continue
            out.append(
                {
                    "dataset": ds,
                    "csv": str(csv.relative_to(ROOT)),
                    "start": start,
                    "end": end,
                    "label": label,
                    "fps": fps,
                }
            )
    return out


def build_real_xy(
    window_sec: float = WINDOW_SEC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict]]:
    """Return window features, labels, video ids, feature names, and row metadata."""
    segs = load_real_segments()
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    xs, ys, groups, kept = [], [], [], []
    names = list(FEATURE_NAMES)
    for s in segs:
        key = s["csv"]
        if key not in cache:
            cache[key] = load_trajectory_csv(ROOT / key)
        pos, vel = cache[key]
        a, b = s["start"], min(s["end"], pos.shape[0])
        Xw, starts = window_feature_matrix(
            pos[a:b],
            vel[a:b],
            window_sec=window_sec,
            fps=s["fps"],
            names=names,
        )
        for i, row in enumerate(Xw):
            xs.append(row)
            ys.append(s["label"])
            groups.append(s["dataset"])
            kept.append({**s, "window_start": int(a + starts[i])})
    if not xs:
        return (
            np.zeros((0, len(names))),
            np.array([], dtype=object),
            np.array([], dtype=object),
            names,
            [],
        )
    return (
        np.vstack(xs),
        np.array(ys, dtype=object),
        np.array(groups, dtype=object),
        names,
        kept,
    )
