"""Confidence calibration analysis for image-level alert scores.

Detectors are often over/under-confident. For a monitoring system that raises
alerts on a threshold, we want the confidence to *mean* something. This module
provides:

    * Expected Calibration Error (ECE) and a reliability diagram,
    * confidence histograms split by outcome (TP / FP / FN) to inform threshold
      selection.

Inputs are image-level: a probability-like score per image and a binary label
(hazard present or not). This is well-defined and robust, unlike box-level
calibration which depends on IoU matching choices.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import FIGURES_DIR, ensure_dir


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> tuple[float, dict]:
    """Compute ECE and per-bin stats for a reliability diagram.

    Returns ``(ece, bins)`` where ``bins`` holds arrays for plotting.
    """
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)

    accs, confs, counts, centers = [], [], [], []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score > lo) & (y_score <= hi) if i > 0 else (y_score >= lo) & (y_score <= hi)
        count = int(mask.sum())
        if count:
            acc = float(y_true[mask].mean())
            conf = float(y_score[mask].mean())
            ece += (count / n) * abs(acc - conf)
        else:
            acc = conf = 0.0
        accs.append(acc)
        confs.append(conf)
        counts.append(count)
        centers.append((lo + hi) / 2)

    bins = {
        "edges": edges.tolist(),
        "centers": centers,
        "accuracy": accs,
        "confidence": confs,
        "count": counts,
    }
    return float(ece), bins


def plot_reliability_diagram(bins: dict, ece: float, out_path: str | Path) -> Path:
    """Save a reliability diagram (accuracy vs. confidence)."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    centers = np.asarray(bins["centers"])
    accs = np.asarray(bins["accuracy"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.bar(centers, accs, width=1.0 / len(centers) * 0.9, alpha=0.7, label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("empirical accuracy (hazard present)")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_confidence_histograms(
    scores_present: np.ndarray,
    scores_absent: np.ndarray,
    out_path: str | Path,
    title: str = "Alert-score distribution",
) -> Path:
    """Overlay score histograms for positive vs. negative images.

    The overlap between the two distributions is exactly the region where
    threshold choice trades recall against precision.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores_present, bins=25, range=(0, 1), alpha=0.6, label="hazard present", density=True)
    ax.hist(scores_absent, bins=25, range=(0, 1), alpha=0.6, label="no hazard", density=True)
    ax.set_xlabel("alert confidence score")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    prefix: str = "alert",
    figures_dir: str | Path = FIGURES_DIR,
    n_bins: int = 10,
) -> dict:
    """Full calibration pass: ECE + reliability diagram + score histograms."""
    figures_dir = Path(figures_dir)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    ece, bins = expected_calibration_error(y_true, y_score, n_bins=n_bins)
    rel_path = plot_reliability_diagram(bins, ece, figures_dir / f"{prefix}_reliability.png")
    hist_path = plot_confidence_histograms(
        y_score[y_true == 1],
        y_score[y_true == 0],
        figures_dir / f"{prefix}_score_hist.png",
        title=f"{prefix} score distribution",
    )
    return {
        "ece": ece,
        "bins": bins,
        "reliability_figure": str(rel_path),
        "histogram_figure": str(hist_path),
    }
