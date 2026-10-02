"""Train XGBoost motion classifier on simulated features."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.features.dataset import build_sim_xy, manifest_has_transitions
from src.labels import is_transition, label_set, ordered_labels

ROOT = Path(__file__).resolve().parents[2]

PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 7, 8, 9],
    "learning_rate": [0.001,0.005,0.01,0.025, 0.05, 0.075, 0.1],
    "n_estimators": [100,200, 300, 400, 500],
}


def _count_by_kind(labels: np.ndarray) -> dict[str, int]:
    base = int(sum(not is_transition(y) for y in labels))
    trans = int(sum(is_transition(y) for y in labels))
    return {"baseline": base, "transition": trans, "total": int(len(labels))}


def _macro_f1(y_true, y_pred) -> float:
    present = np.unique(y_true)
    return float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0))


def _inv_class_weights(y: np.ndarray) -> np.ndarray:
    """Weight each sample by 1/n_class so every class contributes equally."""
    _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv].astype(np.float64)


def train_classifier(
    out_dir: Path | None = None,
    mode: str = "frame",
    manifest_path: Path | None = None,
    sim_root: Path | None = None,
    include_transitions: bool | None = None,
    stable_only: bool = False,
) -> dict:
    out_dir = out_dir or (ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_path or (ROOT / "sim_datasets" / "manifest.json")
    sim_root = sim_root or manifest_path.parent
    if include_transitions is None:
        include_transitions = manifest_has_transitions(manifest_path) and not stable_only

    xy_kw = dict(
        manifest_path=manifest_path,
        sim_root=sim_root,
        mode=mode,
        include_transitions=include_transitions,
        stable_only=stable_only,
    )
    X_tr, y_tr, feat_names = build_sim_xy(split="train", **xy_kw)
    X_te, y_te, _ = build_sim_xy(split="test", **xy_kw)
    if len(X_tr) == 0 or len(X_te) == 0:
        X_sim, y_sim, feat_names = build_sim_xy(split=None, **xy_kw)
        if len(X_sim) == 0:
            raise RuntimeError("No training samples — run generate_sims.py first")
        X_search, y_search_raw = X_sim, y_sim
        cv: PredefinedSplit | StratifiedKFold = StratifiedKFold(
            n_splits=min(5, len(np.unique(y_sim))),
            shuffle=True,
            random_state=42,
        )
        n_train, n_test = int(len(X_sim)), 0
        score_name = "sim CV macro-F1"
    else:
        X_sim = np.vstack([X_tr, X_te])
        y_sim = np.concatenate([y_tr, y_te])
        X_search, y_search_raw = X_sim, y_sim
        cv = PredefinedSplit(np.r_[np.full(len(X_tr), -1), np.zeros(len(X_te), dtype=int)])
        n_train, n_test = int(len(X_tr)), int(len(X_te))
        score_name = "sim test macro-F1"

    classes = label_set(include_transitions=include_transitions, stable_only=stable_only)
    present_labels = set(y_sim)
    eval_labels = ordered_labels(lab for lab in classes if lab in present_labels)

    le = LabelEncoder()
    le.fit(list(classes))
    yt = le.transform(y_sim)
    y_search = le.transform(y_search_raw)
    w_sim = _inv_class_weights(yt)
    if n_test:
        w_search = np.concatenate(
            [_inv_class_weights(le.transform(y_tr)), np.ones(len(X_te), dtype=np.float64)]
        )
    else:
        w_search = w_sim

    n_candidates = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    print(
        f"sim={len(X_sim)}  train={n_train}  test={n_test}  "
        f"classes={len(eval_labels)}  grid={n_candidates} configs (score={score_name})"
    )
    print("param grid:", PARAM_GRID)

    search = GridSearchCV(
        XGBClassifier(random_state=42, n_jobs=1),
        PARAM_GRID,
        scoring=make_scorer(_macro_f1),
        cv=cv,
        n_jobs=-1,
        refit=False,
        verbose=3,
    )
    search.fit(X_search, y_search, sample_weight=w_search)
    model = XGBClassifier(**search.best_params_, random_state=42, n_jobs=-1)
    model.fit(X_sim, yt, sample_weight=w_sim)
    print(f"best_params={search.best_params_}  {score_name}={search.best_score_:.4f}")

    report: dict = {
        "n_train": n_train or int(len(X_sim)),
        "n_test": n_test,
        "include_transitions": include_transitions,
        "stable_only": stable_only,
        "n_classes": len(eval_labels),
        "train_counts": _count_by_kind(y_sim),
        "features": feat_names,
        "best_params": search.best_params_,
        "best_sim_macro_f1": float(search.best_score_),
        "sim_test_macro_f1": float(search.best_score_),
    }

    model_path = out_dir / "classifier.joblib"
    metrics_path = out_dir / "sim_test_metrics.json"
    joblib.dump(
        {
            "model": model,
            "label_encoder": le,
            "feature_names": feat_names,
            "best_params": search.best_params_,
        },
        model_path,
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"wrote {model_path}")
    print(f"wrote {metrics_path}")
    print(f"sim_macro_f1={report['best_sim_macro_f1']:.3f}")
    return report
