# Model Card — D-Fire Wildfire Detector

## Model details

- **Task:** object detection (2 classes: `smoke`, `fire`).
- **Architecture:** YOLOv8m (Ultralytics).
- **Initialization:** COCO-pretrained weights (transfer learning).
- **Input:** RGB images, letterboxed to 640px.
- **Output:** bounding boxes with per-class confidence, collapsed to an alert
  state (`none` / `smoke` / `fire` / `fire_and_smoke`).
- **Framework:** PyTorch + Ultralytics.

## Intended use

- Early-warning fire/smoke detection from fixed cameras, drones, or uploaded
  imagery, as a **decision-support** tool with a human in the loop.
- **Not** a certified safety device; do not use as the sole trigger for
  life-safety actions.

## Training configuration

See `configs/train_yolo_main.yaml`. Highlights: cosine LR with warmup, mixed
precision, early stopping, conservative HSV, no vertical flip, mosaic disabled
for the final epochs.

## Evaluation (test split)

Reported from the Kaggle production run (`yolov8m`, `yolo_main`, imgsz 640,
balanced thresholds `smoke=0.25` / `fire=0.25`):

| Metric                | Value |
| --------------------- | ----- |
| mAP@50-95             | 0.456 |
| mAP@50                | 0.795 |
| AP@50-95 (smoke)      | 0.528 |
| AP@50-95 (fire)       | 0.384 |
| Alert recall          | 0.978 |
| Alert precision       | 0.982 |
| Alert PR-AUC          | 0.996 |
| Category accuracy     | 0.944 |
| Calibration ECE       | 0.128 |

## Operating points

Confidence thresholds are per-class and selected on the **validation** PR
curves, then reported on **test** (`configs/inference.yaml`). This run used
the default balanced point (`0.25` / `0.25`), which matched the values written
into `configs/inference.yaml`:

- **high_recall** — maximize catch rate (early warning); more false alarms.
- **balanced** — default deployment trade-off (**used for the numbers above**).
- **high_precision** — minimize false alarms; may miss faint/small events.

**Why recall is weighted heavily:** a missed fire (false negative) can cause
irreversible damage, whereas a false positive mainly costs operator attention.
Smoke recall is emphasized because smoke often precedes visible flames.

## Limitations

- Struggles on very small/distant or heavily occluded targets.
- Smoke vs. cloud/fog and fire vs. sunset/lights remain the dominant error
  modes (see `reports/error_analysis.md`).
- RGB-only, single-frame: no thermal, weather, or temporal context.
- Trained on D-Fire's domain distribution; expect degradation on very different
  cameras/geographies without fine-tuning.

## Ethical considerations

- False negatives carry safety risk; keep human oversight.
- Avoid deployment claims beyond the evaluated domain.
- Respect the D-Fire dataset license and cite the authors.
