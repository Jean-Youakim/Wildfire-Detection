"""Draw ground-truth and predicted boxes, build comparison grids and galleries.

Uses OpenCV for drawing (fast, no font dependencies) and Matplotlib only for
assembling multi-image grids. Colors come from :data:`src.data.dfire.CLASS_COLORS`
so every artifact in the project is visually consistent.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.dfire import (
    CLASS_COLORS,
    CLASS_NAMES,
    YoloBox,
    label_path_for_image,
    parse_label_file,
    yolo_to_pixel,
)
from src.utils.paths import ensure_dir


def _read_rgb(image_path: str | Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int, int, float | None]],
    thickness: int = 2,
) -> np.ndarray:
    """Draw boxes on an RGB image (returns a copy).

    Each box is ``(cls, x1, y1, x2, y2, conf_or_None)`` in pixel coordinates.
    """
    out = image.copy()
    h = out.shape[0]
    for cls, x1, y1, x2, y2, conf in boxes:
        color = CLASS_COLORS.get(int(cls), (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = CLASS_NAMES.get(int(cls), str(cls))
        if conf is not None:
            label = f"{label} {conf:.2f}"
        font_scale = max(0.4, min(1.0, h / 800))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y_text = max(0, y1 - 4)
        cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 2, y_text), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 1, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _gt_boxes_to_pixels(gt: list[YoloBox], w: int, h: int) -> list[tuple]:
    out = []
    for b in gt:
        x1, y1, x2, y2 = yolo_to_pixel(b, w, h)
        out.append((b.cls, x1, y1, x2, y2, None))
    return out


def save_prediction_overlay(
    image_path: str | Path,
    pred_boxes: list[tuple[int, int, int, int, int, float]],
    out_path: str | Path,
    show_ground_truth: bool = True,
) -> Path:
    """Save a single image with predictions (and optionally GT) overlaid.

    When ``show_ground_truth`` is set, GT boxes are drawn on a left panel and
    predictions on a right panel for easy side-by-side comparison.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    image = _read_rgb(image_path)
    h, w = image.shape[:2]

    pred_panel = draw_boxes(image, pred_boxes)

    if show_ground_truth:
        gt, _ = parse_label_file(label_path_for_image(image_path))
        gt_panel = draw_boxes(image, _gt_boxes_to_pixels(gt, w, h))
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(gt_panel)
        axes[0].set_title("Ground truth")
        axes[1].imshow(pred_panel)
        axes[1].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        cv2.imwrite(str(out_path), cv2.cvtColor(pred_panel, cv2.COLOR_RGB2BGR))
    return out_path


def make_comparison_grid(
    panels: list[tuple[np.ndarray, str]],
    out_path: str | Path,
    ncols: int = 3,
    suptitle: str | None = None,
) -> Path:
    """Assemble a grid of ``(image, caption)`` panels and save it."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    n = len(panels)
    if n == 0:
        raise ValueError("No panels to plot.")
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    # axes may be longer than panels (grid padding), so strict=False.
    for ax, (img, caption) in zip(axes, panels, strict=False):
        ax.imshow(img)
        ax.set_title(caption, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
