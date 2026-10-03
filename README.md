# School Motion Classifier

XGBoost classifier of 2D fish-school motion from five collective order parameters. Training uses the annotated real trajectories only.

## Behaviours

| Label | Description |
| --- | --- |
| `traveling` | Common heading; high translational order |
| `milling` | Circulation around the school centroid |
| `shoaling` | Cohesion with little global heading |
| `expansion` | Outward radial organization |
| `compaction` | Inward radial organization |

Directed transitions are separate classes (`traveling_to_milling`, `expansion_to_compaction`, …). The label set is the five behaviours plus all 20 `a_to_b` pairs (25 names). Classes with no windows yet stay in the list so later annotations reuse the same names. Older annotation strings (`traveling_polarized`, `polarized`, `burst`, `e+`, `contraction`, …) resolve through [`annotations/_label_aliases.json`](annotations/_label_aliases.json).

## Features

Each sample is the mean of five per-frame order parameters over a non-overlapping **0.5 s** window (15 frames at 29.97 fps). Leftover frames shorter than 0.5 s are dropped.

- **φ_trans** — magnitude of the mean unit heading (moving fish only).
- **φ_tan**, **φ_rad^±** — anisotropy-corrected correlations: center `r̂` and `v̂` by their school means, then `φ_tan = |∑ (r' × v')_z| / D` and `φ_rad^± = (∑ r' · v') / D`.
- **φ_tan^unsigned** — fraction of centroid-relative kinetic energy in the tangential direction. Clockwise and counterclockwise both add.
- **φ_local** — mean, over fish with at least one neighbor, of the magnitude of the mean unit heading of up to **5** nearest neighbors inside radius **90** (focal fish excluded).

Degenerate frames: zero speed omits that fish from heading averages; a fish at the centroid is omitted from `r̂`; a vanishing correlation denominator, no centroid-relative motion, or no moving neighbor inside radius 90 yield 0 for the corresponding feature.

## Layout

```
src/io.py          # trajectory CSV / H5 / annotation JSON
src/features/      # order parameters, 0.5 s windows, dataset builder
src/classify/      # train / predict
src/labels.py      # canonical names + aliases
annotations/       # per-video segment labels
schooling-datasets/# trajectories (r = x,y and v = px,py)
scripts/           # CLI
results/           # classifier.joblib + cv_metrics.json
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

Leave-one-video-out over the 10 dataset ids. Trees use `max_depth=6`, `learning_rate=0.05`, `n_estimators=500`. Sample weights are `n / (n_classes · n_class)` so each class contributes equally without dropping below XGBoost’s `min_child_weight`. Reported scores are the pooled out-of-fold predictions. A model is then refit on all windows.

```bash
python scripts/train_classifier.py
python scripts/train_classifier.py --no-transitions
python scripts/plot_confusion.py
```

`--no-transitions` (`--stable-only`) drops every `a_to_b` window so training and leave-one-video-out use only `traveling`, `milling`, `shoaling`, `expansion`, and `compaction`.

Outputs:

- `results/classifier.joblib` — refit model, label encoder, full 25-name list
- `results/cv_metrics.json` — per-fold and pooled OOF metrics
- `results/cv_confusion.png`

## Predict

```bash
python scripts/predict_trajectory.py path/to/traj.csv results/classifier.joblib out.csv
```

Each 0.5 s window is labelled and that label is repeated across the window’s frames.
