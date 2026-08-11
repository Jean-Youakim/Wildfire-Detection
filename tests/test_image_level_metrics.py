"""Unit tests for the image-level alert metrics and category confusion."""

from __future__ import annotations

import numpy as np

from src.evaluation.image_level_metrics import (
    category_confusion,
    compute_image_level_metrics,
    predicted_category,
)


def _perfect_inputs():
    """4 images: none, smoke_only, fire_only, both — perfectly predicted."""
    y_true = {"smoke": np.array([0, 1, 0, 1]), "fire": np.array([0, 0, 1, 1])}
    y_score = {"smoke": np.array([0.0, 0.9, 0.1, 0.8]), "fire": np.array([0.0, 0.1, 0.9, 0.9])}
    return y_true, y_score


def test_perfect_predictions():
    y_true, y_score = _perfect_inputs()
    result = compute_image_level_metrics(y_true, y_score, {"smoke": 0.5, "fire": 0.5})
    for name in ("smoke", "fire"):
        m = result.per_class[name]
        assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
        assert m["roc_auc"] == 1.0
    assert result.alert["recall"] == 1.0 and result.alert["precision"] == 1.0
    assert result.alert["n_images"] == 4


def test_threshold_changes_predictions():
    y_true, y_score = _perfect_inputs()
    # A very high threshold suppresses every prediction -> recall 0.
    result = compute_image_level_metrics(y_true, y_score, {"smoke": 0.99, "fire": 0.99})
    assert result.per_class["smoke"]["recall"] == 0.0
    assert result.per_class["fire"]["recall"] == 0.0


def test_single_class_labels_do_not_crash():
    y_true = {"smoke": np.array([0, 0]), "fire": np.array([0, 0])}
    y_score = {"smoke": np.array([0.1, 0.2]), "fire": np.array([0.0, 0.3])}
    result = compute_image_level_metrics(y_true, y_score, {"smoke": 0.5, "fire": 0.5})
    assert np.isnan(result.per_class["smoke"]["roc_auc"])
    assert np.isnan(result.per_class["smoke"]["pr_auc"])


def test_predicted_category():
    assert predicted_category(False, False) == "none"
    assert predicted_category(True, False) == "smoke_only"
    assert predicted_category(False, True) == "fire_only"
    assert predicted_category(True, True) == "fire_and_smoke"


def test_category_confusion_perfect_is_diagonal():
    y_true, y_score = _perfect_inputs()
    out = category_confusion(y_true, y_score, {"smoke": 0.5, "fire": 0.5})
    matrix = np.asarray(out["matrix"])
    assert matrix.sum() == 4
    assert np.trace(matrix) == 4
    assert out["accuracy"] == 1.0


def test_category_confusion_counts_mistakes():
    y_true = {"smoke": np.array([1]), "fire": np.array([0])}
    y_score = {"smoke": np.array([0.1]), "fire": np.array([0.9])}  # smoke misread as fire
    out = category_confusion(y_true, y_score, {"smoke": 0.5, "fire": 0.5})
    matrix = np.asarray(out["matrix"])
    labels = out["labels"]
    r, c = labels.index("smoke_only"), labels.index("fire_only")
    assert matrix[r, c] == 1
    assert out["accuracy"] == 0.0
