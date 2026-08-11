"""Image-level ("alert") metrics derived from detection outputs.

A wildfire *monitoring* system ultimately answers a per-image/per-frame
question: *should we raise an alert, and for what?* These metrics translate
box-level detections into that operational view:

    * per-class presence detection (smoke / fire): ROC-AUC, PR-AUC, and
      precision/recall/F1 at a chosen confidence threshold,
    * an overall "any alert" score,
    * a 4-way category confusion matrix
      (none / smoke_only / fire_only / fire_and_smoke).

All functions operate on plain NumPy arrays so they are framework-agnostic and
unit-testable, independent of Ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.dfire import (
    CATEGORIES,
    CATEGORY_BOTH,
    CATEGORY_FIRE,
    CATEGORY_NONE,
    CATEGORY_SMOKE,
    CLASS_NAMES,
)


@dataclass
class ImageLevelResult:
    """Container for image-level metrics (JSON-serializable via ``as_dict``)."""

    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    alert: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"per_class": self.per_class, "alert": self.alert, "thresholds": self.thresholds}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC that degrades gracefully when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_image_level_metrics(
    y_true_present: dict[str, np.ndarray],
    y_score: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> ImageLevelResult:
    """Compute per-class and overall alert metrics.

    Args:
        y_true_present: ``{class_name: bool array [N]}`` — ground-truth presence.
        y_score: ``{class_name: float array [N]}`` — model confidence that the
            class is present in the image (e.g. max box confidence for that
            class, or 0.0 if none predicted).
        thresholds: ``{class_name: float}`` decision thresholds.

    Returns:
        An :class:`ImageLevelResult`.
    """
    result = ImageLevelResult(thresholds=dict(thresholds))

    for name in CLASS_NAMES.values():
        yt = np.asarray(y_true_present[name]).astype(int)
        ys = np.asarray(y_score[name]).astype(float)
        thr = float(thresholds.get(name, 0.25))
        yp = (ys >= thr).astype(int)
        result.per_class[name] = {
            "roc_auc": _safe_auc(yt, ys),
            "pr_auc": _safe_ap(yt, ys),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support_positive": int(yt.sum()),
        }

    # Overall "alert" = any hazard present. Score = max over class scores.
    names = list(CLASS_NAMES.values())
    true_any = np.zeros_like(np.asarray(y_true_present[names[0]]), dtype=int)
    score_any = np.zeros_like(np.asarray(y_score[names[0]]), dtype=float)
    pred_any = np.zeros_like(true_any, dtype=int)
    for name in names:
        yt = np.asarray(y_true_present[name]).astype(int)
        ys = np.asarray(y_score[name]).astype(float)
        thr = float(thresholds.get(name, 0.25))
        true_any = np.maximum(true_any, yt)
        score_any = np.maximum(score_any, ys)
        pred_any = np.maximum(pred_any, (ys >= thr).astype(int))

    result.alert = {
        "roc_auc": _safe_auc(true_any, score_any),
        "pr_auc": _safe_ap(true_any, score_any),
        "precision": float(precision_score(true_any, pred_any, zero_division=0)),
        "recall": float(recall_score(true_any, pred_any, zero_division=0)),
        "f1": float(f1_score(true_any, pred_any, zero_division=0)),
        "support_positive": int(true_any.sum()),
        "n_images": int(len(true_any)),
    }
    return result


def predicted_category(smoke_present: bool, fire_present: bool) -> str:
    """Map presence booleans to one of :data:`CATEGORIES`."""
    if smoke_present and fire_present:
        return CATEGORY_BOTH
    if fire_present:
        return CATEGORY_FIRE
    if smoke_present:
        return CATEGORY_SMOKE
    return CATEGORY_NONE


def category_confusion(
    y_true_present: dict[str, np.ndarray],
    y_score: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict:
    """Build a 4x4 category confusion matrix (rows = true, cols = predicted)."""
    smoke_true = np.asarray(y_true_present["smoke"]).astype(bool)
    fire_true = np.asarray(y_true_present["fire"]).astype(bool)
    smoke_pred = np.asarray(y_score["smoke"]) >= float(thresholds.get("smoke", 0.25))
    fire_pred = np.asarray(y_score["fire"]) >= float(thresholds.get("fire", 0.25))

    index = {c: i for i, c in enumerate(CATEGORIES)}
    matrix = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
    for st, ft, sp, fp in zip(smoke_true, fire_true, smoke_pred, fire_pred, strict=True):
        r = index[predicted_category(bool(st), bool(ft))]
        c = index[predicted_category(bool(sp), bool(fp))]
        matrix[r, c] += 1

    accuracy = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else float("nan")
    return {"labels": list(CATEGORIES), "matrix": matrix.tolist(), "accuracy": accuracy}
