# Data

The **D-Fire dataset is NOT committed** to this repository (it is large and has
its own license). This folder documents how the code expects data to be laid
out.

## Expected layout

```
data/
├── train/
│   ├── images/   # *.jpg
│   └── labels/   # *.txt  (YOLO format, one file per image)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Label format (YOLO)

Each `labels/<name>.txt` contains zero or more rows:

```
<class_id> <x_center> <y_center> <width> <height>
```

- Coordinates are **normalized** to `[0, 1]` relative to image width/height.
- **Class IDs (verified against the D-Fire spec and the files themselves):**
  - `0` = **smoke**
  - `1` = **fire**
- **Negative images** (no fire, no smoke) have an **empty** `.txt` file (or no
  file). These are essential for reducing false alarms and are kept on purpose.

## Getting the data

- Official source: https://github.com/gaiasd/DFireDataset
- Ready-to-use YOLO version on Kaggle: `sayedgamal99/smoke-fire-detection-yolo`

### Local
Download and arrange it to match the layout above (this repo already contains a
small verified subset in that layout for development).

### Kaggle
Attach the dataset to your notebook; it will mount under
`/kaggle/input/<dataset-slug>/`. Pass that root via `--data-root` to the
scripts (see the top-level `README.md`).

## Citation

> P. V. A. B. de Venâncio, A. C. Lisboa, A. V. Barbosa. "An automatic fire
> detection system based on deep convolutional neural networks for low-power,
> resource-constrained devices." *Neural Computing and Applications*, 2022.
