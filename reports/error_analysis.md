# Error Analysis

> Populate the galleries and counts by running `notebooks/03_error_analysis.ipynb`
> (or the underlying library functions). This document describes the taxonomy of
> failures we track and how to act on each.

## Method

We evaluate at the image ("alert") level and the box level:

- **False positive (FP):** the model raises an alert on a `none` image.
- **False negative (FN):** the model misses a real fire/smoke image.
- Box-level errors are inspected via GT-vs-prediction overlays
  (`src/visualization/plot_predictions.py`).

Galleries are written to:

- `outputs/figures/false_positives.png`
- `outputs/figures/false_negatives.png`

## Failure taxonomy (what to look for)

| Cause                         | Typical class | Mitigation |
| ----------------------------- | ------------- | ---------- |
| Cloud / fog mistaken as smoke | smoke FP      | hard-negative mining; keep negatives |
| Sunset / lamp / reflection    | fire FP       | conservative HSV; hard negatives |
| Distant / tiny flame missed   | fire FN       | train at a larger image size; lower fire conf |
| Faint / low-contrast smoke    | smoke FN      | contrast-robust augmentation |
| Night / backlit scenes        | both          | ensure night samples represented |
| Partial occlusion             | both          | random-erasing augmentation |
| Ambiguous / noisy labels      | both          | manual re-check; document |

## Threshold sensitivity

Use `evaluate_detector.py --tune-thresholds` to see how precision/recall trade
off per class. Record the chosen operating point and its rationale in the model
card.

## Action items (template)

1. Top FP source: `TBD` -> `TBD` mitigation.
2. Top FN source: `TBD` -> `TBD` mitigation.
3. Retrain / adjust thresholds; re-evaluate on the held-out test split.
