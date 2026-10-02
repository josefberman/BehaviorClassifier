"""CSV / JSON / HDF5 IO for schooling trajectory tables."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CSV_SUFFIXES = {".csv"}
H5_SUFFIXES = {".h5", ".hdf5"}


def trajectory_to_dataframe(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> pd.DataFrame:
    """positions, velocities: (T, N, 2)."""
    t, n, _ = positions.shape
    cols: dict[str, np.ndarray] = {"frame": np.arange(t, dtype=np.int32)}
    for i in range(n):
        cols[f"fish{i}_x"] = positions[:, i, 0]
        cols[f"fish{i}_y"] = positions[:, i, 1]
        cols[f"fish{i}_vx"] = velocities[:, i, 0]
        cols[f"fish{i}_vy"] = velocities[:, i, 1]
    return pd.DataFrame(cols)


def save_trajectory_csv(
    path: Path,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_to_dataframe(positions, velocities).to_csv(path, index=False)


def _decode_h5_str(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _h5_string_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype(encoding="utf-8"))


def load_trajectory_table(path: Path) -> pd.DataFrame:
    """Load a trajectory table from CSV or H5 (`columns` + `data`)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in CSV_SUFFIXES:
        return pd.read_csv(path)
    if suffix in H5_SUFFIXES:
        with h5py.File(path, "r") as f:
            if "columns" not in f or "data" not in f:
                raise ValueError(f"{path} must contain datasets 'columns' and 'data'")
            cols = [_decode_h5_str(c) for c in f["columns"][:]]
            df = pd.DataFrame(np.asarray(f["data"]), columns=cols)
            reserved = {"columns", "data"}
            for name in f.keys():
                if name in reserved or name in df.columns:
                    continue
                ds = f[name]
                if getattr(ds, "shape", ()) == (len(df),):
                    df[name] = [_decode_h5_str(v) for v in ds[:]]
            return df
    raise ValueError(f"Unsupported trajectory type {path.suffix!r}; expected .csv, .h5, or .hdf5")


def save_trajectory_table(path: Path, df: pd.DataFrame) -> None:
    """Write a trajectory table as CSV or H5, matching the output suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in CSV_SUFFIXES:
        df.to_csv(path, index=False)
        return
    if suffix in H5_SUFFIXES:
        numeric = df.select_dtypes(include=[np.number])
        extra = [c for c in df.columns if c not in numeric.columns]
        with h5py.File(path, "w") as f:
            f.create_dataset("columns", data=_h5_string_array([str(c) for c in numeric.columns]))
            f.create_dataset("data", data=numeric.to_numpy(dtype=np.float64))
            for col in extra:
                f.create_dataset(str(col), data=_h5_string_array([str(v) for v in df[col].tolist()]))
        return
    raise ValueError(f"Unsupported trajectory type {path.suffix!r}; expected .csv, .h5, or .hdf5")


def pos_vel_from_table(df: pd.DataFrame, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Return positions, velocities as (T, N, 2). Missing vx/vy are finite-differenced."""
    fish_ids = sorted(
        {int(c[4:].split("_")[0]) for c in df.columns if c.startswith("fish") and c.endswith("_x")}
    )
    if not fish_ids:
        raise ValueError("No fish{i}_x columns found in trajectory table")
    t = len(df)
    n = len(fish_ids)
    pos = np.zeros((t, n, 2), dtype=np.float64)
    vel = np.zeros((t, n, 2), dtype=np.float64)
    have_vel = all(f"fish{fid}_vx" in df.columns and f"fish{fid}_vy" in df.columns for fid in fish_ids)
    for i, fid in enumerate(fish_ids):
        pos[:, i, 0] = df[f"fish{fid}_x"].to_numpy(dtype=np.float64)
        pos[:, i, 1] = df[f"fish{fid}_y"].to_numpy(dtype=np.float64)
        if have_vel:
            vel[:, i, 0] = df[f"fish{fid}_vx"].to_numpy(dtype=np.float64)
            vel[:, i, 1] = df[f"fish{fid}_vy"].to_numpy(dtype=np.float64)
    if not have_vel and t >= 2:
        vel[1:] = np.diff(pos, axis=0) * float(fps)
        vel[0] = vel[1]
    return pos, vel


def load_trajectory_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return positions, velocities as (T, N, 2)."""
    return pos_vel_from_table(pd.read_csv(path))


def mmss_to_frame(mmss: str, fps: float) -> int:
    parts = mmss.strip().split(":")
    if len(parts) == 2:
        m, s = int(parts[0]), float(parts[1])
        total = m * 60 + s
    elif len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        total = h * 3600 + m * 60 + s
    else:
        raise ValueError(f"Bad timestamp: {mmss!r}")
    return int(round(total * fps))


def load_motion_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
