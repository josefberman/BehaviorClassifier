"""Evaluate trained classifier on real annotated segments."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from src.features.dataset import build_real_xy
from src.labels import is_transition, ordered_labels

ROOT = Path(__file__).resolve().parents[2]


def _count_by_kind(labels: np.ndarray) -> dict[str, int]:
    base = int(sum(not is_transition(y) for y in labels))
    trans = int(sum(is_transition(y) for y in labels))
    return {"baseline": base, "transition": trans, "total": int(len(labels))}


def eval_real(
    model_path: Path | None = None,
    out_dir: Path | None = None,
    include_transitions: bool | None = None,
    stable_only: bool = False,
) -> dict:
    out_dir = out_dir or (ROOT / "results")
    model_path = model_path or (out_dir / "classifier.joblib")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]
    model_classes = [str(c) for c in le.classes_]

    if include_transitions is None:
        include_transitions = any(is_transition(c) for c in model_classes)
    if stable_only:
        include_transitions = False

    X_all, y_all, _, _meta_all = build_real_xy(
        include_transitions=include_transitions,
        stable_only=stable_only,
        mode="frame",
    )
    known = set(le.classes_)
    known_mask = np.array([yi in known for yi in y_all])
    X, y = X_all[known_mask], y_all[known_mask]

    eval_labels = ordered_labels(set(y))
    report: dict = {
        "n_real": int(len(y)),
        "include_transitions": include_transitions,
        "stable_only": stable_only,
        "n_classes_in_model": int(len(le.classes_)),
        "label_counts": {lab: int(np.sum(y == lab)) for lab in eval_labels},
        "segment_counts": _count_by_kind(y),
        "n_skipped_unknown_label": int(np.sum(~known_mask)),
    }

    if len(X) == 0:
        report["error"] = "no real segments"
        return report

    pred = le.inverse_transform(model.predict(X))
    report["accuracy"] = float(np.mean(pred == y))
    report["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    report["macro_f1"] = float(
        f1_score(y, pred, average="macro", labels=eval_labels, zero_division=0)
    )
    report["classification_report"] = classification_report(
        y, pred, labels=eval_labels, zero_division=0, output_dict=True
    )
    report["confusion_matrix"] = confusion_matrix(y, pred, labels=eval_labels).tolist()
    report["confusion_labels"] = eval_labels

    base_mask = np.array([not is_transition(yi) for yi in y])
    trans_mask = np.array([is_transition(yi) for yi in y])
    if base_mask.any():
        base_labels = ordered_labels({yi for yi in y[base_mask]})
        report["baseline_accuracy"] = float(np.mean(pred[base_mask] == y[base_mask]))
        report["baseline_balanced_accuracy"] = float(
            balanced_accuracy_score(y[base_mask], pred[base_mask])
        )
        report["baseline_macro_f1"] = float(
            f1_score(
                y[base_mask],
                pred[base_mask],
                average="macro",
                labels=base_labels,
                zero_division=0,
            )
        )
    if trans_mask.any():
        trans_labels = ordered_labels({yi for yi in y[trans_mask]})
        report["transition_accuracy"] = float(np.mean(pred[trans_mask] == y[trans_mask]))
        report["transition_balanced_accuracy"] = float(
            balanced_accuracy_score(y[trans_mask], pred[trans_mask])
        )
        report["transition_macro_f1"] = float(
            f1_score(
                y[trans_mask],
                pred[trans_mask],
                average="macro",
                labels=trans_labels,
                zero_division=0,
            )
        )

    with open(out_dir / "real_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report
