# AnytimeRR

Implementation of **Subject-block calibrated adaptive observation for PPG respiratory-rate estimation**. A causal prefix predictor estimates rate and residual scale; one conformal score per subject covers four episodes and five prefixes before a quality-gated stopping policy selects an output.

## Install

Use Python 3.10 or newer:

```bash
python -m venv .venv
python -m pip install -e .
```

Activate the virtual environment before installation. PyTorch uses the CPU by default; pass `--device cuda` for a compatible GPU installation.

## Prepare recordings

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

```bash
anytimerr prepare --manifest data/manifest.json --output data/prepared
anytimerr train --data data/prepared --epochs 40 --output runs/train
anytimerr evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

Preparation extracts four nonoverlapping 64-second episodes per subject. Each target uses only the last completed valid interbreath interval at the corresponding 16, 24, 32, 48 or 64-second endpoint. Intervals outside 1.4–10 seconds are ineligible; preparation rejects an incomplete block instead of selecting episodes by prediction error. Review reference eligibility before finalizing the manifest.

## Comparisons and ablations

Run the same frozen checkpoint and partitions with separate output directories:

```bash
anytimerr evaluate --policy fixed16 --output runs/fixed16
anytimerr evaluate --policy fixed32 --output runs/fixed32
anytimerr evaluate --policy fixed64 --output runs/fixed64
anytimerr evaluate --predictor spectral --policy fixed32 --output runs/spectral32
anytimerr evaluate --policy scale --output runs/scale_stopping
anytimerr evaluate --calibration marginal --output runs/marginal
anytimerr evaluate --calibration pooled --output runs/pooled
anytimerr evaluate --no-gate --output runs/no_gate
anytimerr evaluate --constant-scale --output runs/constant_scale
anytimerr evaluate --width 1.0 --output runs/width1
anytimerr evaluate --width 3.0 --output runs/width3
```

`--alpha` selects the conformal error level; `--width` sets the allowable interval half-width in breaths/min. `--min-cycles` and `--concentration` define observable respiratory evidence. Select these values using development subjects, then freeze them. `--epochs`, `--lr` and `--seed` control fitting. `--checkpoint`, `--data` and `--device` may be passed to every evaluation.

`block` takes the maximum standardized residual over every episode and prefix within each calibration subject. `marginal` calibrates each prefix using pooled episodes; `pooled` pools all episode/prefix scores. The latter two are controls with different calibration units. A rank beyond the available scores produces unbounded intervals and adaptive withholding. `fixed16/32/64` always emit at the declared horizon; `scale` stops using the residual scale directly.

## Outputs

Training writes `model.pt` and `history.json`. Evaluation writes:

- `metrics.json`: emitted MAE/RMSE in breaths/min, emitted interval coverage, whole-block joint coverage, output fraction and mean observation time.
- `episodes.json`: subject, episode, selected horizon, estimate, reference, emission and coverage decisions, and interval half-width.
- `rates.png`: emitted estimates against their selected-prefix targets.

MAE, RMSE and emitted coverage use emitted episodes. Joint coverage uses all episode/prefix targets, including candidates never selected. Observation time includes withheld episodes at 64 seconds. With no emitted estimates, conditional metrics are `null`; withholding does not become a zero error.

## Implementation choices

The compact encoder uses left-padded convolutions and prefix-only normalization; no future episode samples enter a prediction. Positive scales are trained with the paper's absolute-residual/scale plus log-scale objective and equal weight per subject. The cycle gate uses spectral concentration in 0.1–0.7 Hz and the implied number of observed cycles, calculated from the current prefix only. The spectral baseline uses that band directly. It is a spectral-modulation comparison; the compact learned encoder is not labeled RRWaveNet or a Smart Fusion implementation. The fixed-horizon learned controls share this encoder to isolate observation and calibration choices. Input conversion from proprietary CapnoBase formats is performed before manifest preparation.
