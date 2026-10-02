"""Nested leave-one-video-out XGBoost trainer on annotated real windows."""

from __future__ import annotations

import json
from collections import Counter
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.classify.eval import macro_f1, metrics as eval_metrics
from src.features.dataset import build_real_xy
from src.features.windows import WINDOW_SEC
from src.labels import ALL_LABELS, ordered_labels

ROOT = Path(__file__).resolve().parents[2]

PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
}

DEFAULT_PARAMS = {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 200}


def _grid() -> list[dict]:
    keys = list(PARAM_GRID)
    return [dict(zip(keys, vals)) for vals in product(*PARAM_GRID.values())]


def _inv_class_weights(y: np.ndarray) -> np.ndarray:
    _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv].astype(np.float64)


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


def _inner_score(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict,
    inner_videos: np.ndarray,
) -> float:
    scores: list[float] = []
    for held in inner_videos:
        tr = groups != held
        te = groups == held
        if not tr.any() or not te.any():
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        model, le = _fit_xgb(X[tr], y[tr], params)
        pred = _predict(model, le, X[te])
        scores.append(macro_f1(y[te], pred))
    if not scores:
        return float("-inf")
    return float(np.mean(scores))


def _select_params(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict:
    videos = np.unique(groups)
    if len(videos) < 2:
        return dict(DEFAULT_PARAMS)
    best_params = dict(DEFAULT_PARAMS)
    best_score = float("-inf")
    for params in _grid():
        score = _inner_score(X, y, groups, params, videos)
        if score > best_score:
            best_score = score
            best_params = dict(params)
    return best_params


def train_classifier(
    out_dir: Path | None = None,
    window_sec: float = WINDOW_SEC,
) -> dict:
    out_dir = out_dir or (ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, feat_names, _meta = build_real_xy(window_sec=window_sec)
    if len(X) == 0:
        raise RuntimeError("No training windows — check annotations and schooling-datasets")

    videos = np.unique(groups)
    print(
        f"windows={len(X)}  videos={len(videos)}  "
        f"classes_present={len(np.unique(y))}  grid={len(_grid())} configs"
    )

    oof_true: list[str] = []
    oof_pred: list[str] = []
    fold_reports: list[dict] = []
    fold_params: list[dict] = []

    for held in videos:
        tr = groups != held
        te = groups == held
        if not te.any() or len(np.unique(y[tr])) < 2:
            print(f"skip video {held}: empty test or <2 train classes")
            continue
        params = _select_params(X[tr], y[tr], groups[tr])
        model, le = _fit_xgb(X[tr], y[tr], params)
        pred = _predict(model, le, X[te])
        y_te = y[te]
        fold = {
            "held_out": str(held),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "best_params": params,
            "macro_f1": macro_f1(y_te, pred),
        }
        fold_reports.append(fold)
        fold_params.append(params)
        oof_true.extend(y_te.tolist())
        oof_pred.extend(pred.tolist())
        print(f"held_out={held}  n_test={fold['n_test']}  macro_f1={fold['macro_f1']:.4f}  params={params}")

    y_true = np.array(oof_true, dtype=object)
    y_pred = np.array(oof_pred, dtype=object)
    pooled = eval_metrics(y_true, y_pred)

    param_counts = Counter(tuple(sorted(p.items())) for p in fold_params)
    if param_counts:
        winner = param_counts.most_common(1)[0][0]
        final_params = dict(winner)
    else:
        final_params = dict(DEFAULT_PARAMS)

    final_model, final_le = _fit_xgb(X, y, final_params)
    present_labels = ordered_labels(np.unique(y))

    report = {
        "n_windows": int(len(X)),
        "n_videos": int(len(videos)),
        "window_sec": float(window_sec),
        "features": feat_names,
        "all_labels": list(ALL_LABELS),
        "fitted_labels": [str(c) for c in final_le.classes_],
        "n_classes_fitted": int(len(final_le.classes_)),
        "label_counts": {lab: int(np.sum(y == lab)) for lab in ALL_LABELS},
        "best_params": final_params,
        "fold_params": fold_params,
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
            "all_labels": list(ALL_LABELS),
            "fitted_labels": present_labels,
            "best_params": final_params,
            "window_sec": float(window_sec),
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
