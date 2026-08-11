"""Validate D-Fire dataset integrity before training.

Checks, per split:
    * every image has a readable file (optionally opened with Pillow),
    * every label row is well-formed YOLO (5 numeric fields),
    * class ids are in {0 (smoke), 1 (fire)},
    * coordinates are within [0, 1] and boxes have positive size,
    * detects orphan labels (label without image) and duplicate boxes,
    * counts negative images (empty/missing label files) as valid.

Outputs a JSON report and a human-readable summary. Exits non-zero if any
*hard* error is found (so it can gate a CI/training pipeline).

Usage:
    python -m src.data.validate_dataset --data-root data
    python -m src.data.validate_dataset --data-root /kaggle/input/xxx --check-images
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.dfire import (
    CLASS_NAMES,
    SEVERITY_ERROR,
    SPLITS,
    YoloBox,
    iter_split,
    parse_label_file,
)
from src.utils import get_logger, resolve_data_root
from src.utils.paths import METRICS_DIR, ensure_dir, split_dirs

logger = get_logger("validate")


def _duplicate_boxes(boxes: list[YoloBox]) -> int:
    """Count exact-duplicate boxes (same class + rounded coordinates)."""
    seen: set[tuple] = set()
    dupes = 0
    for b in boxes:
        key = (
            b.cls,
            round(b.x_center, 5),
            round(b.y_center, 5),
            round(b.width, 5),
            round(b.height, 5),
        )
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
    return dupes


def validate_split(data_root: Path, split: str, check_images: bool) -> dict:
    """Validate a single split; return a structured result dict."""
    images_dir, labels_dir = split_dirs(data_root, split)
    result: dict = {
        "split": split,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "n_images": 0,
        "n_negatives": 0,
        "n_boxes": 0,
        "class_box_counts": {name: 0 for name in CLASS_NAMES.values()},
        "errors": [],       # hard (structural) -> blocks training
        "fixable": [],      # geometric -> auto-clipped/dropped by trainer
        "warnings": [],     # non-fatal bookkeeping (duplicates, orphans)
    }

    if not images_dir.is_dir():
        result["errors"].append(f"missing images dir: {images_dir}")
        return result

    image_stems: set[str] = set()

    for img_path, lbl_path in iter_split(images_dir):
        result["n_images"] += 1
        image_stems.add(img_path.stem)

        if check_images:
            if not _image_readable(img_path):
                result["errors"].append(f"unreadable image: {img_path.name}")

        boxes, issues = parse_label_file(lbl_path)
        for issue in issues:
            bucket = "errors" if issue.severity == SEVERITY_ERROR else "fixable"
            result[bucket].append(f"{lbl_path.name}: {issue}")

        if not boxes:
            result["n_negatives"] += 1

        dupes = _duplicate_boxes(boxes)
        if dupes:
            result["warnings"].append(f"{lbl_path.name}: {dupes} duplicate box(es)")

        for b in boxes:
            result["n_boxes"] += 1
            if b.cls in CLASS_NAMES:
                result["class_box_counts"][CLASS_NAMES[b.cls]] += 1

    # Orphan labels: a .txt with no corresponding image.
    if labels_dir.is_dir():
        for lbl in labels_dir.glob("*.txt"):
            if lbl.stem not in image_stems:
                result["warnings"].append(f"orphan label (no image): {lbl.name}")

    return result


def _image_readable(path: Path) -> bool:
    """Return True if the image opens and verifies with Pillow."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


def validate_dataset(data_root: Path, check_images: bool = False) -> dict:
    """Validate every split and return the aggregated report."""
    logger.info("Validating dataset at: %s", data_root)
    report: dict = {"data_root": str(data_root), "splits": {}, "totals": {}}

    total_errors = 0
    totals = Counter()
    class_totals = Counter()

    for split in SPLITS:
        res = validate_split(data_root, split, check_images)
        report["splits"][split] = res
        total_errors += len(res["errors"])
        totals["images"] += res["n_images"]
        totals["negatives"] += res["n_negatives"]
        totals["boxes"] += res["n_boxes"]
        for name, count in res["class_box_counts"].items():
            class_totals[name] += count

        logger.info(
            "  %-5s: %5d images | %5d negatives | %6d boxes | %d errors | %d fixable | %d warnings",
            split,
            res["n_images"],
            res["n_negatives"],
            res["n_boxes"],
            len(res["errors"]),
            len(res["fixable"]),
            len(res["warnings"]),
        )

    total_fixable = sum(len(report["splits"][s]["fixable"]) for s in report["splits"])
    report["totals"] = {
        "images": totals["images"],
        "negatives": totals["negatives"],
        "boxes": totals["boxes"],
        "class_box_counts": dict(class_totals),
        "n_errors": total_errors,
        "n_fixable": total_fixable,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the D-Fire dataset.")
    parser.add_argument(
        "--data-root", default=None, help="Dataset root (auto-detected if omitted)."
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Also open every image with Pillow (slower, catches corrupt files).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the JSON report (default: outputs/metrics/validation.json).",
    )
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    report = validate_dataset(data_root, check_images=args.check_images)

    out_path = Path(args.output) if args.output else METRICS_DIR / "validation.json"
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote validation report -> %s", out_path)

    n_errors = report["totals"]["n_errors"]
    n_fixable = report["totals"]["n_fixable"]
    if n_fixable:
        logger.warning(
            "%d auto-fixable geometry issue(s) found (out-of-range/zero-area boxes). "
            "These are clipped/dropped at train time; see the report for details.",
            n_fixable,
        )
    if n_errors:
        logger.error("Validation FAILED with %d hard (structural) error(s).", n_errors)
        return 1
    logger.info(
        "Validation PASSED. Total images: %d | boxes: %d | negatives: %d",
        report["totals"]["images"],
        report["totals"]["boxes"],
        report["totals"]["negatives"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
