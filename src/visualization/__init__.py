"""Visualization: prediction overlays and comparison grids."""

from .plot_predictions import (
    draw_boxes,
    make_comparison_grid,
    save_prediction_overlay,
)

__all__ = ["draw_boxes", "save_prediction_overlay", "make_comparison_grid"]
