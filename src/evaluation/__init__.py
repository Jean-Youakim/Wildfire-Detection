"""Evaluation: detection metrics, image-level alert metrics, and calibration."""

from .image_level_metrics import (
    ImageLevelResult,
    category_confusion,
    compute_image_level_metrics,
    predicted_category,
)

__all__ = [
    "ImageLevelResult",
    "compute_image_level_metrics",
    "category_confusion",
    "predicted_category",
]
