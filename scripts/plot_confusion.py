#!/usr/bin/env python3
"""Plot the leave-one-video-out confusion matrix from cv_metrics.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labels import ordered_labels

_CMAP = LinearSegmentedColormap.from_list(
    "mid_blues",
    plt.cm.Blues(np.linspace(0.0, 0.55, 256)),
)


def _reorder(cm: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    desired = ordered_labels(labels)
    idx = [labels.index(lab) for lab in desired]
    return cm[np.ix_(idx, idx)], desired


def _row_percent(cm: np.ndarray) -> np.ndarray:
    totals = cm.sum(axis=1, keepdims=True)
    pct = np.zeros_like(cm, dtype=float)
    np.divide(cm * 100.0, totals, out=pct, where=totals > 0)
    return pct


def _cell_text(pct: float, count: int) -> str:
    if pct <= 0:
        pct_s = "0%"
    elif abs(pct - 100.0) < 0.05:
        pct_s = "100%"
    else:
        pct_s = f"{pct:.1f}%"
    return f"{pct_s}\n({count})"


def plot_cm(path: Path, out: Path, title: str) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    block = data.get("oof", data)
    if "confusion_matrix" not in block:
        print(f"No confusion_matrix in {path}")
        return
    cm = np.asarray(block["confusion_matrix"], dtype=float)
    labels = list(block.get("confusion_labels", [str(i) for i in range(cm.shape[0])]))
    cm, labels = _reorder(cm, labels)
    pct = _row_percent(cm)
    tick = [lab.replace("_", " ") for lab in labels]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * n), max(7.0, 0.4 * n)))
    ax.imshow(pct, cmap=_CMAP, vmin=0.0, vmax=100.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick, rotation=45, ha="right")
    ax.set_yticklabels(tick)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fontsize = 8 if n <= 10 else 5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                _cell_text(pct[i, j], int(cm[i, j])),
                ha="center",
                va="center",
                color="black",
                fontsize=fontsize,
            )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    cv_json = ROOT / "results" / "cv_metrics.json"
    if not cv_json.exists():
        print(f"Missing {cv_json}")
        return
    plot_cm(cv_json, ROOT / "results" / "cv_confusion.png", "Leave-one-video-out confusion")


if __name__ == "__main__":
    main()
