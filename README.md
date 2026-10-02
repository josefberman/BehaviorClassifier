# School Motion Classifier

Simulate five fish-school behaviours with one Couzin-zone dynamical model, extract collective order parameters, train a classifier on simulations, and evaluate on held-out sims plus manually annotated real trajectories.

## Behaviours

Behavior differences come from a 13-parameter set (`r_r, r_o, r_a, w_r, w_o, w_a, w_tan, w_rad, sigma_theta, s_0, sigma_s, omega_max, a_max`). Expansion/compaction are signed radial steering (`w_rad`); milling is tangential steering (`w_tan`).

| Canonical label       | Short      | Description                                        |
| --------------------- | ---------- | -------------------------------------------------- |
| `traveling_polarized` | tpol       | High `w_o`, low `sigma_theta`; net translation     |
| `milling`             | milling    | `w_tan > 0`; rotation about the school centroid |
| `shoaling`            | shoaling   | Weak `w_o`, larger `sigma_theta`; low polarization |
| `expansion_burst`     | expansion  | `w_rad > 0`; outward radial tendency           |
| `compaction`          | compaction | `w_rad < 0`; inward radial tendency            |

`fountain_evasion` remains as an unused YAML stub and is not part of the five-class training set.

## Simulator

One model for all behaviors. Social interactions use exclusive Couzin zones (`d < r_r` repulsion, `r_r ≤ d < r_o` orientation, `r_o ≤ d < r_a` attraction). Heading noise `epsilon_w_i` and speed noise `epsilon_a_i` are sampled i.i.d. `Normal(0,1)` each step and are not YAML parameters. Arena, `dt`, `burn_in`, and `record_frames` are simulation metadata.

Classifier inputs: instantaneous per-frame **Φ_trans**, **Φ_tan**, **Φ_rad^±**, **Φ_tan^unsigned**, and **Φ_local** (5 nearest neighbors). Training and evaluation inverse-weight classes by frame count.

## Layout

```
src/sim/           # Couzin-zone simulator + IO
src/features/      # order parameters, windows, dataset builders
src/classify/      # train / eval
configs/behaviors/ # frozen YAML parameter sets
scripts/           # CLI entry points
sim_datasets/      # generated trajectories + manifest.json
schooling-datasets/# real trajectories
annotations/       # real segment labels (test)
results/           # models + metrics
```



## Setup

```bash
pip install -r requirements.txt
```



## Generate simulations

Generate sims × 5 behaviours × group sizes with train/test split.

```bash
python scripts/generate_sims.py --n-jobs 8
# quick check:
python scripts/generate_sims.py --smoke --n-jobs 4
```



### Publication figures / video

Clean white-background stills (300 dpi PNG) and optional MP4/GIF. Fish are dark discs with short heading ticks; circular arena outline only.

```bash
# While generating (stills for seed 0 only + video):
python scripts/generate_sims.py --smoke --render --video --render-seeds 0 --n-jobs 2

# From existing sim_datasets (manuscript set: one clip per behaviour):
python scripts/render_sims.py --manuscript --video

# Selected clips:
python scripts/render_sims.py --n-values 30 --seeds 0 --video
```

Outputs land in `sim_datasets/.../renders/` or `results/figures/<behavior>/`.

## Calibrate / inspect signatures

Summarizes mean/std of the five per-frame order-parameter features per behavior. Default source is generated sims (`sim_datasets/manifest.json`); use `--source real` for manual annotations.

```bash
python scripts/calibrate_baselines.py
python scripts/calibrate_baselines.py --source real
python scripts/calibrate_baselines.py --source both
```



## Train & evaluate

```bash
# Default: include transition clips/segments when manifest/model supports them
python scripts/train_classifier.py
python scripts/eval_real.py

# Stable states only (tpol, milling, shoaling) — no transition labels
python scripts/train_classifier.py --stable-only
python scripts/eval_real.py --stable-only
```

`--stable-only` is an alias for `--no-transitions`.

Outputs:

- `results/classifier.joblib`
- `results/sim_test_metrics.json`
- `results/real_eval_metrics.json`
- `results/sim_confusion.png` / `results/real_confusion.png`
- `results/calibration_report.json`

```bash
python scripts/plot_confusion.py
```



## Dataset split


| Split | Seeds | Count |
| ----- | ----- | ----- |
| train | 0–79  | 2880  |
| test  | 80–99 | 720   |




## Real data notes

- Label aliases live in `annotations/_label_aliases.json` (`polarized`→`traveling_polarized`, `burst`/`spread`→`expansion_burst`, …).
- Real annotations are class-imbalanced; fountain/burst are rare; **compaction has no real labels** and is excluded from real eval.
- Expect a sim↔real domain gap: the model is trained only on simulations; real metrics are a secondary sanity check.

