"""Exploratory data analysis for the D-Fire dataset.

Computes and visualizes:
    * image counts per split and per image-level category
      (none / smoke_only / fire_only / fire_and_smoke),
    * per-class bounding-box counts,
    * box-size distribution (COCO-style small/medium/large buckets) to expose
      the small-object challenge that drives our high-resolution experiment,
    * boxes-per-image distribution.

Artifacts:
    * outputs/metrics/dataset_stats.json  (machine-readable)
    * outputs/figures/*.png               (plots)
    * reports/dataset_report.md           (human-readable, auto-filled)

Usage:
    python -m src.data.analyze_dataset --data-root data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / Kaggle safe
import matplotlib.pyplot as plt

from src.data.dfire import (
    CATEGORIES,
    CLASS_NAMES,
    SPLITS,
    image_category,
    iter_split,
    parse_label_file,
)
from src.utils import get_logger, resolve_data_root
from src.utils.paths import FIGURES_DIR, METRICS_DIR, REPORTS_DIR, ensure_dir, split_dirs

logger = get_logger("analyze")

# COCO-style area buckets on normalized area (fraction of image area).
# small  : < 0.01  (< ~1% of the image; e.g. distant flames)
# medium : 0.01 - 0.09
# large  : > 0.09
SMALL_MAX = 0.01
MEDIUM_MAX = 0.09


def _area_bucket(norm_area: float) -> str:
    if norm_area < SMALL_MAX:
        return "small"
    if norm_area < MEDIUM_MAX:
        return "medium"
    return "large"


def analyze_split(data_root: Path, split: str) -> dict:
    """Collect raw statistics for one split."""
    images_dir, _ = split_dirs(data_root, split)
    stats: dict = {
        "n_images": 0,
        "categories": {c: 0 for c in CATEGORIES},
        "class_box_counts": {name: 0 for name in CLASS_NAMES.values()},
        "area_buckets": {
            name: {"small": 0, "medium": 0, "large": 0} for name in CLASS_NAMES.values()
        },
        "boxes_per_image": {},  # summary (see below), not the raw per-image list
    }

    counts: list[int] = []
    for _img_path, lbl_path in iter_split(images_dir):
        stats["n_images"] += 1
        boxes, _ = parse_label_file(lbl_path)
        stats["categories"][image_category(boxes)] += 1
        counts.append(len(boxes))
        for b in boxes:
            name = CLASS_NAMES.get(b.cls)
            if name is None:
                continue
            stats["class_box_counts"][name] += 1
            stats["area_buckets"][name][_area_bucket(b.area)] += 1

    stats["boxes_per_image"] = _summarize_counts(counts)
    return stats


def _summarize_counts(counts: list[int]) -> dict:
    """Compact summary of the per-image box-count distribution."""
    if not counts:
        return {"max": 0, "mean": 0.0, "histogram": {}}
    buckets = {"0": 0, "1": 0, "2": 0, "3-5": 0, "6-10": 0, ">10": 0}
    for c in counts:
        if c == 0:
            buckets["0"] += 1
        elif c == 1:
            buckets["1"] += 1
        elif c == 2:
            buckets["2"] += 1
        elif c <= 5:
            buckets["3-5"] += 1
        elif c <= 10:
            buckets["6-10"] += 1
        else:
            buckets[">10"] += 1
    return {
        "max": int(max(counts)),
        "mean": round(sum(counts) / len(counts), 3),
        "histogram": buckets,
    }


def _plot_category_distribution(per_split: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(CATEGORIES))
    width = 0.8 / max(len(per_split), 1)
    for i, (split, s) in enumerate(per_split.items()):
        vals = [s["categories"][c] for c in CATEGORIES]
        ax.bar([xi + i * width for xi in x], vals, width=width, label=split)
    ax.set_xticks([xi + width * (len(per_split) - 1) / 2 for xi in x])
    ax.set_xticklabels(CATEGORIES, rotation=15)
    ax.set_ylabel("images")
    ax.set_title("Image-level category distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_class_boxes(per_split: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(CLASS_NAMES.values())
    x = range(len(names))
    width = 0.8 / max(len(per_split), 1)
    for i, (split, s) in enumerate(per_split.items()):
        vals = [s["class_box_counts"][n] for n in names]
        ax.bar([xi + i * width for xi in x], vals, width=width, label=split)
    ax.set_xticks([xi + width * (len(per_split) - 1) / 2 for xi in x])
    ax.set_xticklabels(names)
    ax.set_ylabel("bounding boxes")
    ax.set_title("Bounding boxes per class by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_area_buckets(totals: dict, out: Path) -> None:
    buckets = ["small", "medium", "large"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(buckets))
    width = 0.35
    for i, name in enumerate(CLASS_NAMES.values()):
        vals = [totals["area_buckets"][name][b] for b in buckets]
        ax.bar([xi + i * width for xi in x], vals, width=width, label=name)
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(buckets)
    ax.set_ylabel("bounding boxes")
    ax.set_title("Box-size distribution (small-object challenge)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _aggregate_totals(per_split: dict[str, dict]) -> dict:
    totals = {
        "n_images": 0,
        "categories": Counter(),
        "class_box_counts": Counter(),
        "area_buckets": {name: Counter() for name in CLASS_NAMES.values()},
    }
    for s in per_split.values():
        totals["n_images"] += s["n_images"]
        totals["categories"].update(s["categories"])
        totals["class_box_counts"].update(s["class_box_counts"])
        for name in CLASS_NAMES.values():
            totals["area_buckets"][name].update(s["area_buckets"][name])
    # make JSON-serializable
    totals["categories"] = dict(totals["categories"])
    totals["class_box_counts"] = dict(totals["class_box_counts"])
    totals["area_buckets"] = {k: dict(v) for k, v in totals["area_buckets"].items()}
    return totals


def _write_markdown_report(data_root: Path, per_split: dict, totals: dict, fig_paths: dict) -> Path:
    lines: list[str] = []
    lines.append("# D-Fire Dataset Report\n")
    lines.append(f"_Auto-generated by `src/data/analyze_dataset.py` from `{data_root}`._\n")
    lines.append("## Class convention\n")
    lines.append("| id | class |")
    lines.append("| -- | ----- |")
    for cid, name in CLASS_NAMES.items():
        lines.append(f"| {cid} | {name} |")
    lines.append("")

    lines.append("## Images per split\n")
    lines.append("| split | images |")
    lines.append("| ----- | ------ |")
    for split, s in per_split.items():
        lines.append(f"| {split} | {s['n_images']} |")
    lines.append(f"| **total** | **{totals['n_images']}** |")
    lines.append("")

    lines.append("## Image-level categories\n")
    header = "| split | " + " | ".join(CATEGORIES) + " |"
    sep = "| " + " | ".join(["---"] * (len(CATEGORIES) + 1)) + " |"
    lines.append(header)
    lines.append(sep)
    for split, s in per_split.items():
        row = [str(s["categories"][c]) for c in CATEGORIES]
        lines.append(f"| {split} | " + " | ".join(row) + " |")
    tot_row = [str(totals["categories"].get(c, 0)) for c in CATEGORIES]
    lines.append("| **total** | " + " | ".join(tot_row) + " |")
    lines.append("")

    lines.append("## Bounding boxes per class\n")
    lines.append("| class | " + " | ".join(per_split.keys()) + " | total |")
    lines.append("| " + " | ".join(["---"] * (len(per_split) + 2)) + " |")
    for name in CLASS_NAMES.values():
        per = [str(per_split[sp]["class_box_counts"][name]) for sp in per_split]
        total = totals["class_box_counts"].get(name, 0)
        lines.append(f"| {name} | " + " | ".join(per) + f" | {total} |")
    lines.append("")

    lines.append("## Box-size distribution (normalized area)\n")
    lines.append("Buckets: small `<1%`, medium `1-9%`, large `>9%` of image area.\n")
    lines.append("| class | small | medium | large |")
    lines.append("| ----- | ----- | ------ | ----- |")
    for name in CLASS_NAMES.values():
        ab = totals["area_buckets"][name]
        lines.append(
            f"| {name} | {ab.get('small', 0)} | {ab.get('medium', 0)} | {ab.get('large', 0)} |"
        )
    lines.append("")

    lines.append("## Figures\n")
    for title, path in fig_paths.items():
        p = Path(path)
        rel = p.relative_to(REPORTS_DIR.parent) if REPORTS_DIR.parent in p.parents else path
        lines.append(f"- **{title}**: `{rel}`")
    lines.append("")

    lines.append("## Notes & risks\n")
    lines.append(
        "- **Negatives matter**: `none` images are kept to suppress false alarms "
        "(clouds, fog, sunsets, lights).\n"
        "- **Small objects**: any sizeable `small` bucket justifies the "
        "high-resolution (832px) experiment for recall on distant fire/smoke.\n"
        "- **Class asymmetry**: report class-wise AP so strong smoke performance "
        "cannot mask weak fire performance (or vice-versa).\n"
        "- **Leakage**: D-Fire frames can come from the same source video; keep "
        "the official split and never move files between splits."
    )
    lines.append("")

    out = REPORTS_DIR / "dataset_report.md"
    ensure_dir(out.parent)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the D-Fire dataset.")
    parser.add_argument(
        "--data-root", default=None, help="Dataset root (auto-detected if omitted)."
    )
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    ensure_dir(FIGURES_DIR)
    ensure_dir(METRICS_DIR)

    per_split: dict[str, dict] = {}
    for split in SPLITS:
        logger.info("Analyzing split: %s", split)
        per_split[split] = analyze_split(data_root, split)

    totals = _aggregate_totals(per_split)

    fig_paths = {
        "Category distribution": FIGURES_DIR / "category_distribution.png",
        "Boxes per class": FIGURES_DIR / "boxes_per_class.png",
        "Box-size distribution": FIGURES_DIR / "box_size_distribution.png",
    }
    _plot_category_distribution(per_split, fig_paths["Category distribution"])
    _plot_class_boxes(per_split, fig_paths["Boxes per class"])
    _plot_area_buckets(totals, fig_paths["Box-size distribution"])

    stats_path = METRICS_DIR / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"data_root": str(data_root), "per_split": per_split, "totals": totals}, f, indent=2
        )
    logger.info("Wrote stats -> %s", stats_path)

    report_path = _write_markdown_report(data_root, per_split, totals, fig_paths)
    logger.info("Wrote report -> %s", report_path)
    logger.info("Figures -> %s", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
