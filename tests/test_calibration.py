"""Unit tests for the ECE computation."""

from __future__ import annotations

import numpy as np

from src.evaluation.calibration import expected_calibration_error


def test_perfectly_calibrated_scores():
    # Scores equal to the empirical positive rate within each bin -> ECE ~ 0.
    y_score = np.array([0.25] * 4 + [0.75] * 4)
    y_true = np.array([1, 0, 0, 0, 1, 1, 1, 0])  # 25% and 75% positive
    ece, bins = expected_calibration_error(y_true, y_score, n_bins=4)
    assert ece < 1e-9
    assert sum(bins["count"]) == len(y_true)


def test_maximally_overconfident():
    # Model says 1.0 on all-negative labels -> ECE = 1.
    y_true = np.zeros(10)
    y_score = np.ones(10)
    ece, _ = expected_calibration_error(y_true, y_score, n_bins=10)
    assert abs(ece - 1.0) < 1e-9


def test_bins_cover_all_samples():
    rng = np.random.default_rng(0)
    y_score = rng.random(500)
    y_true = (rng.random(500) < y_score).astype(int)
    ece, bins = expected_calibration_error(y_true, y_score, n_bins=10)
    assert sum(bins["count"]) == 500
    assert 0.0 <= ece <= 1.0
