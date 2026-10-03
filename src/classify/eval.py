"""Metrics helpers for leave-one-video-out reports."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from src.labels import is_transition, label_set, ordered_labels


def macro_f1(y_true, y_pred) -> float:
    present = ordered_labels(np.unique(y_true))
    if not present:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0))


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    include_transitions: bool = True,
) -> dict:
    classes = label_set(include_transitions=include_transitions)
    present = ordered_labels(set(y_true) | set(y_pred))
    scored = ordered_labels(set(y_true))
    report = {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_pred == y_true)) if len(y_true) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": macro_f1(y_true, y_pred),
        "label_counts": {lab: int(np.sum(y_true == lab)) for lab in classes},
        "classification_report": classification_report(
            y_true, y_pred, labels=scored, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=present).tolist(),
        "confusion_labels": present,
    }
    base_mask = np.array([not is_transition(yi) for yi in y_true])
    trans_mask = np.array([is_transition(yi) for yi in y_true])
    if base_mask.any():
        report["behavior_macro_f1"] = macro_f1(y_true[base_mask], y_pred[base_mask])
    if include_transitions and trans_mask.any():
        report["transition_macro_f1"] = macro_f1(y_true[trans_mask], y_pred[trans_mask])
    return report
