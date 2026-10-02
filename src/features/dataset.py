"""Build feature matrices from simulated and real annotated trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.features.order_params import AGG_FEATURE_NAMES, FEATURE_NAMES
from src.features.windows import (
    feature_dict_to_array,
    frame_feature_matrix,
    segment_feature_vector,
    sliding_window_features,
)
from src.labels import canonicalize, is_transition, label_set, load_aliases
from src.sim.io import load_motion_json, load_trajectory_csv, mmss_to_frame

ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path: Path | None = None) -> list[dict]:
    path = path or (ROOT / "sim_datasets" / "manifest.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _clip_frame_range(a: int, b: int, n_frames: int, *, min_len: int = 2) -> tuple[int, int]:
    a = max(0, min(a, n_frames - min_len))
    b = max(a + min_len, min(b, n_frames))
    return a, b


def _transition_morph_range(entry: dict, n_frames: int) -> tuple[int, int]:
    """Frame slice [start, end) for the parameter-morph phase of a transition clip."""
    morph_start = int(entry["morph_start"])
    morph_end = int(entry.get("morph_end", n_frames))
    return _clip_frame_range(morph_start, morph_end, n_frames)


def _names_for_mode(mode: str) -> list[str]:
    return list(FEATURE_NAMES) if mode == "frame" else list(AGG_FEATURE_NAMES)


def features_from_sim_entry(
    entry: dict,
    mode: str = "segment",
    window_sec: float = 2.0,
    sim_root: Path | None = None,
) -> list[tuple[np.ndarray, str]]:
    """Return list of (x, label).

    Baseline clips use the full recording. Transition clips use only the morph
    window (morph_start .. morph_end) so features match real transition segments.
    mode='frame' emits one row per frame (instantaneous Φ values).
    """
    sim_root = sim_root or (ROOT / "sim_datasets")
    csv_path = Path(entry["csv"])
    if not csv_path.is_absolute():
        csv_path = sim_root / csv_path
    pos, vel = load_trajectory_csv(csv_path)
    fps = float(entry.get("fps", 30.0))
    label = canonicalize(entry["behavior"])

    if entry.get("morph_start") is not None:
        a, b = _transition_morph_range(entry, pos.shape[0])
        pos, vel = pos[a:b], vel[a:b]

    if mode == "frame":
        if entry.get("event_start") is not None:
            a = int(entry["event_start"])
            b = int(entry.get("event_end", min(pos.shape[0], a + 200)))
            a, b = _clip_frame_range(a, b, pos.shape[0])
            pos, vel = pos[a:b], vel[a:b]
        X = frame_feature_matrix(pos, vel, fps=fps)
        return [(row, label) for row in X]

    if mode == "windows":
        feats = sliding_window_features(pos, vel, window_sec=window_sec, fps=fps)
    elif entry.get("event_start") is not None:
        a = int(entry["event_start"])
        b = int(entry.get("event_end", min(pos.shape[0], a + 200)))
        a, b = _clip_frame_range(a, b, pos.shape[0])
        feats = [segment_feature_vector(pos[a:b], vel[a:b], fps=fps)]
    else:
        feats = [segment_feature_vector(pos, vel, fps=fps)]
    return [(feature_dict_to_array(f), label) for f in feats]


def manifest_has_transitions(manifest_path: Path | None = None) -> bool:
    return any("_to_" in e.get("behavior", "") for e in load_manifest(manifest_path))


def build_sim_xy(
    split: str | None = None,
    manifest_path: Path | None = None,
    sim_root: Path | None = None,
    mode: str = "frame",
    include_transitions: bool = True,
    stable_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    manifest_path = manifest_path or (ROOT / "sim_datasets" / "manifest.json")
    sim_root = sim_root or manifest_path.parent
    entries = [e for e in load_manifest(manifest_path) if e.get("valid", True)]
    if split is not None:
        entries = [e for e in entries if e.get("split") == split]
    allowed = set(label_set(include_transitions=include_transitions, stable_only=stable_only))
    kept = []
    for e in entries:
        try:
            lab = canonicalize(e.get("behavior", ""))
        except ValueError:
            continue
        if lab not in allowed:
            continue
        if not include_transitions and is_transition(lab):
            continue
        e = {**e, "behavior": lab}
        kept.append(e)
    entries = kept
    xs, ys = [], []
    for e in entries:
        for x, y in features_from_sim_entry(e, mode=mode, sim_root=sim_root):
            xs.append(x)
            ys.append(y)
    names = _names_for_mode(mode)
    if not xs:
        return np.zeros((0, len(names))), np.array([], dtype=object), names
    return np.vstack(xs), np.array(ys, dtype=object), names


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
        for seg in ann["segments"]:
            try:
                label = canonicalize(seg["label"], aliases)
            except ValueError:
                continue
            start = mmss_to_frame(seg["start"], fps)
            end = mmss_to_frame(seg["end"], fps)
            if end - start < int(0.5 * fps):
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
    min_frames: int = 15,
    include_transitions: bool = True,
    stable_only: bool = False,
    mode: str = "frame",
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    segs = load_real_segments()
    if not include_transitions:
        segs = [s for s in segs if not is_transition(s["label"])]
    xs, ys, kept = [], [], []
    for s in segs:
        pos, vel = load_trajectory_csv(ROOT / s["csv"])
        a, b = s["start"], min(s["end"], pos.shape[0])
        if b - a < min_frames:
            continue
        if mode == "frame":
            X = frame_feature_matrix(pos[a:b], vel[a:b], fps=s["fps"])
            for row in X:
                xs.append(row)
                ys.append(s["label"])
                kept.append(s)
        else:
            feat = segment_feature_vector(pos[a:b], vel[a:b], fps=s["fps"])
            xs.append(feature_dict_to_array(feat))
            ys.append(s["label"])
            kept.append(s)
    names = _names_for_mode(mode)
    if not xs:
        return np.zeros((0, len(names))), np.array([], dtype=object), names, []
    return np.vstack(xs), np.array(ys, dtype=object), names, kept
