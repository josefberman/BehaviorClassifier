"""Nested leave-one-video-out XGBoost trainer on annotated real windows."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.classify.eval import macro_f1, metrics as eval_metrics
from src.features.dataset import build_real_xy
from src.features.windows import WINDOW_SEC
from src.labels import label_set, ordered_labels

ROOT = Path(__file__).resolve().parents[2]

XGB_PARAMS = {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 500}


def _inv_class_weights(y: np.ndarray) -> np.ndarray:
    """Balanced weights: n / (n_classes * n_class).

    Raw 1/n_class is too small for the majority class (traveling ≈ 3e-5 at
    0.1 s windows) and falls under XGBoost's default min_child_weight=1, so
    those samples never form leaves and the model never predicts traveling.
    """
    n = float(len(y))
    _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    return n / (len(counts) * counts[inv].astype(np.float64))


def _fit_xgb(X: np.ndarray, y: np.ndarray, params: dict) -> tuple[XGBClassifier, LabelEncoder]:
    le = LabelEncoder()
    yt = le.fit_transform(y)
    n_classes = int(len(le.classes_))
    model = XGBClassifier(
        **params,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(X, yt, sample_weight=_inv_class_weights(yt))
    return model, le


def _predict(model: XGBClassifier, le: LabelEncoder, X: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict(X))
    return le.inverse_transform(raw)


def train_classifier(
    out_dir: Path | None = None,
    window_sec: float = WINDOW_SEC,
    include_transitions: bool = True,
) -> dict:
    out_dir = out_dir or (ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = label_set(include_transitions=include_transitions)

    X, y, groups, feat_names, _meta = build_real_xy(
        window_sec=window_sec,
        include_transitions=include_transitions,
    )
    if len(X) == 0:
        raise RuntimeError("No training windows — check annotations and schooling-datasets")

    videos = np.unique(groups)
    params = dict(XGB_PARAMS)
    print(
        f"windows={len(X)}  videos={len(videos)}  "
        f"classes_present={len(np.unique(y))}  transitions={include_transitions}  "
        f"params={params}"
    )

    oof_true: list[str] = []
    oof_pred: list[str] = []
    fold_reports: list[dict] = []

    for held in videos:
        tr = groups != held
        te = groups == held
        if not te.any() or len(np.unique(y[tr])) < 2:
            print(f"skip video {held}: empty test or <2 train classes")
            continue
        model, le = _fit_xgb(X[tr], y[tr], params)
        pred = _predict(model, le, X[te])
        y_te = y[te]
        fold = {
            "held_out": str(held),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "params": params,
            "macro_f1": macro_f1(y_te, pred),
        }
        fold_reports.append(fold)
        oof_true.extend(y_te.tolist())
        oof_pred.extend(pred.tolist())
        pred_counts = {lab: int(np.sum(pred == lab)) for lab in ordered_labels(np.unique(y_te))}
        print(
            f"held_out={held}  n_test={fold['n_test']}  "
            f"macro_f1={fold['macro_f1']:.4f}  pred={pred_counts}"
        )

    y_true = np.array(oof_true, dtype=object)
    y_pred = np.array(oof_pred, dtype=object)
    pooled = eval_metrics(y_true, y_pred, include_transitions=include_transitions)

    final_model, final_le = _fit_xgb(X, y, params)
    present_labels = ordered_labels(np.unique(y))

    report = {
        "n_windows": int(len(X)),
        "n_videos": int(len(videos)),
        "window_sec": float(window_sec),
        "include_transitions": include_transitions,
        "features": feat_names,
        "all_labels": list(classes),
        "fitted_labels": [str(c) for c in final_le.classes_],
        "n_classes_fitted": int(len(final_le.classes_)),
        "label_counts": {lab: int(np.sum(y == lab)) for lab in classes},
        "params": params,
        "folds": fold_reports,
        "oof": pooled,
    }

    model_path = out_dir / "classifier.joblib"
    metrics_path = out_dir / "cv_metrics.json"
    joblib.dump(
        {
            "model": final_model,
            "label_encoder": final_le,
            "feature_names": feat_names,
            "all_labels": list(classes),
            "fitted_labels": present_labels,
            "params": params,
            "window_sec": float(window_sec),
            "include_transitions": include_transitions,
        },
        model_path,
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"wrote {model_path}")
    print(f"wrote {metrics_path}")
    print(f"oof_macro_f1={pooled['macro_f1']:.3f}  balanced_accuracy={pooled['balanced_accuracy']:.3f}")
    return report
