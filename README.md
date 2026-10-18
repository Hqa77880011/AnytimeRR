# AnytimeRR

Implementation of **Subject-block calibrated adaptive observation for PPG respiratory-rate estimation**. A causal prefix predictor estimates rate and residual scale; one conformal score per subject covers four episodes and five prefixes before a quality-gated stopping policy selects an output.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Obtain PPG and annotated breath events from the [BIDMC PPG and Respiration Dataset](https://physionet.org/content/bidmc/1.0.0/) or the CapnoBase respiratory-rate benchmark. Export one NumPy archive per subject with `ppg` (one-dimensional measured signal), scalar `fs` (Hz), and `breath_times` (annotated breath-event timestamps in seconds from recording start). Keep the signal at its recorded sampling rate. Do not replace breath events with an unseen future-window average.

Create `data/manifest.json`, with paths relative to that manifest:

```json
[
  {"subject": "subject01", "split": "train", "file": "raw/subject01.npz", "starts": [0, 96, 192, 288]},
  {"subject": "subject02", "split": "dev", "file": "raw/subject02.npz", "starts": [0, 96, 192, 288]},
  {"subject": "subject03", "split": "cal", "file": "raw/subject03.npz", "starts": [0, 96, 192, 288]},
  {"subject": "subject04", "split": "test", "file": "raw/subject04.npz", "starts": [0, 96, 192, 288]}
]
```

Use all selected subjects in your actual manifest. Assign complete subjects to mutually exclusive partitions before creating windows. The example demonstrates the schema; it is too small for finite 90% block calibration. Freeze starts, partitions and annotation eligibility before fitting. Ten independent calibration subjects allow a finite 90% rank; a 95% rank needs at least nineteen.



Preparation extracts four nonoverlapping 64-second episodes per subject. Each target uses only the last completed valid interbreath interval at the corresponding 16, 24, 32, 48 or 64-second endpoint. Intervals outside 1.4–10 seconds are ineligible; preparation rejects an incomplete block instead of selecting episodes by prediction error. Review reference eligibility before finalizing the manifest.

```bash
anytimerr prepare --manifest data/manifest.json --output data/prepared
```

## Training and inference

```bash
anytimerr train --data data/prepared --epochs 40 --output runs/train --seed 42
```

## Evaluation

```bash
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

## Comparisons and ablations

```bash
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --policy fixed16 --output runs/fixed16
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --policy fixed32 --output runs/fixed32
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --policy fixed64 --output runs/fixed64
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --predictor spectral --policy fixed32 --output runs/spectral32
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --policy scale --output runs/scale_stopping
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --calibration marginal --output runs/marginal
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --calibration pooled --output runs/pooled
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --no-gate --output runs/no_gate
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --constant-scale --output runs/constant_scale
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --width 1.0 --output runs/width1
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --width 3.0 --output runs/width3
```

Evaluate each fitted training control with its own checkpoint.

## Outputs

Training writes `model.pt` and `history.json`. Evaluation writes:

- `metrics.json`: emitted MAE/RMSE in breaths/min, emitted interval coverage, whole-block joint coverage, output fraction and mean observation time.
- `episodes.json`: subject, episode, selected horizon, estimate, reference, emission and coverage decisions, and interval half-width.
- `rates.png`: emitted estimates against their selected-prefix targets.

MAE, RMSE and emitted coverage use emitted episodes. Joint coverage uses all episode/prefix targets, including candidates never selected. Observation time includes withheld episodes at 64 seconds. With no emitted estimates, conditional metrics are `null`; withholding does not become a zero error.
