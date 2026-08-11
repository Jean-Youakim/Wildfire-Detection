# Wildfire Detection — Fire & Smoke Early Warning

An AI system that finds **fire** and **smoke** in photos and turns them into an
**alert**. Built on the [D-Fire dataset](https://github.com/gaiasd/DFireDataset)
using YOLOv8, with an emphasis on the metric that actually matters for a
monitoring system: *did we catch the fire?*

Lebanon loses forest and farmland to wildfires every summer, and detection is
often delayed until a fire is already large. This project is a step toward
camera-based early warning: given an image from a fixed camera, drone, or phone,
it flags fire and smoke and shows where they are.

> **Class convention:** `0 = smoke`, `1 = fire`.

---

## Quick start

```bash
pip install -r requirements.txt
```

Then use the single entry point:

```bash
python run.py demo                 # open the web demo (easiest way to try it)
python run.py detect photo.jpg     # check one image from the terminal
python run.py train                # train the model (needs GPU + dataset images)
python run.py evaluate             # measure model quality on the test set
```

`python run.py demo` is the one to use for a live demonstration.

> **You need a trained model file** at
> `outputs/weights/yolo_main/weights/best.pt` before `demo`, `detect`, or
> `evaluate` will work. Training happens on Kaggle (see `notebooks/`); download
> the resulting `best.pt` into that path.

---

## How it works

```text
D-Fire images + labels
        ↓
  validate + analyze          (src/data/)
        ↓
  train YOLOv8m               (src/training/  ← configs/train_yolo_main.yaml)
        ↓
  best.pt weights             (outputs/weights/)
        ↓
  evaluate                    (src/evaluation/  → mAP + alert metrics)
        ↓
  WildfireDetector            (src/inference/detector.py)
        ↓
  web demo / CLI              (app/streamlit_app.py, run.py detect)
```

The single most important class is **`WildfireDetector`** in
`src/inference/detector.py`. It loads the trained model, applies per-class
confidence thresholds, and collapses the detected boxes into one alert state:
`none`, `smoke`, `fire`, or `fire_and_smoke`. Both the web demo and the CLI use
it, so they always behave identically.

---

## Repository structure

```
run.py            # single entry point: demo / detect / train / evaluate
configs/          # dataset, training, and inference settings (YAML)
data/             # dataset layout docs (images are NOT committed)
notebooks/        # Kaggle workflow: 01 exploration, 02 training, 03 errors, 04 demo
src/
  data/           # dfire.py (class/label conventions), validation, analysis
  training/       # config-driven YOLO training
  evaluation/     # detection metrics, alert metrics, calibration
  inference/      # WildfireDetector + image prediction
  visualization/  # drawing boxes on images
  utils/          # paths, config loading, device, seeding, logging
reports/          # dataset report, model card, error analysis
outputs/          # weights, metrics, figures (generated)
tests/            # unit tests (no GPU or dataset needed)
app/              # Streamlit demo
```

---

## Getting the data

The dataset is not committed. Use the official repo or the ready-made Kaggle
YOLO version (`sayedgamal99/smoke-fire-detection-yolo`), arranged as:

```
<root>/{train,val,test}/images/*.jpg
<root>/{train,val,test}/labels/*.txt
```

Point the tools at it with `--data-root <root>` or `export DFIRE_ROOT=<root>`.
Locally `./data` is auto-detected; on Kaggle `/kaggle/input/*` is scanned.

---

## Results (test split)

Trained on Kaggle: YOLOv8m, 640px, balanced thresholds (`smoke=0.25`,
`fire=0.25`).

| Metric | Value |
| --- | --- |
| mAP@50 | 0.795 |
| mAP@50-95 | 0.456 |
| AP@50-95 (smoke) | 0.528 |
| AP@50-95 (fire) | 0.384 |
| **Alert recall** | **0.978** |
| **Alert precision** | **0.982** |
| Alert PR-AUC | 0.996 |
| Category accuracy | 0.944 |

Full numbers: `outputs/metrics/eval_test.json`. Interpretation and limitations:
`reports/model_card.md`.

---

## Why alert metrics, not just mAP

Wildfire monitoring is asymmetric: a **missed fire** can be catastrophic, while
a **false alarm** mostly costs attention. So beyond standard detection mAP, this
project measures the operational question directly — *should we raise an alert?*

- per-class ROC-AUC / PR-AUC / precision / recall / F1 at image level
- a 4-way confusion matrix (`none` / `smoke_only` / `fire_only` / `fire_and_smoke`)
- **confidence calibration** (ECE), so a threshold means something in practice

Three named operating modes let an operator pick the trade-off without retuning
anything (`configs/inference.yaml`):

| Mode | Threshold | Use case |
| --- | --- | --- |
| `high_recall` | 0.10 | Early warning; accepts more false alarms |
| `balanced` | 0.25 | Default |
| `high_precision` | 0.50 | Minimize false alarms |

---

## Tests

Label parsing, alert metrics, and calibration are covered by unit tests that
need no GPU and no dataset:

```bash
pytest
```

---

## Limitations

- Trained on D-Fire, which is **not** Lebanese imagery. It is a public proxy;
  fine-tuning on local camera data is the natural next step.
- Single-frame RGB only — no thermal, weather, or video context.
- Small or distant flames and faint smoke remain the hardest cases.
- Decision-support only; not a certified safety device.

---

## Acknowledgements

Dataset: **D-Fire** by Venâncio et al. — please cite their work (see
[`data/README.md`](data/README.md)). Detection framework: Ultralytics YOLO.
This repository's code is MIT-licensed (`LICENSE`); the dataset retains its own
license.
