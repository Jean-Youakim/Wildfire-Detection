"""Shared D-Fire domain library.

This module is the single source of truth for the dataset's conventions so the
validation, analysis, splitting, evaluation and inference code never disagree
about e.g. class ordering.

D-Fire class convention (verified against the label files AND the official
spec):

    0 -> smoke
    1 -> fire

Labels are YOLO-format: ``<class_id> <x_center> <y_center> <width> <height>``
with all coordinates normalized to ``[0, 1]``. Negative images (no fire/smoke)
have an empty label file or no label file at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# --- Class convention -------------------------------------------------------
SMOKE_ID: int = 0
FIRE_ID: int = 1
CLASS_NAMES: dict[int, str] = {SMOKE_ID: "smoke", FIRE_ID: "fire"}

# BGR-friendly RGB tuples used consistently across all visualizations.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    SMOKE_ID: (30, 144, 255),   # dodger blue for smoke
    FIRE_ID: (220, 20, 60),     # crimson for fire
}

SPLITS: tuple[str, ...] = ("train", "val", "test")
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Image-level category labels derived from the set of boxes present.
CATEGORY_NONE = "none"
CATEGORY_SMOKE = "smoke_only"
CATEGORY_FIRE = "fire_only"
CATEGORY_BOTH = "fire_and_smoke"
CATEGORIES: tuple[str, ...] = (CATEGORY_NONE, CATEGORY_SMOKE, CATEGORY_FIRE, CATEGORY_BOTH)


@dataclass(frozen=True)
class YoloBox:
    """A single normalized YOLO bounding box."""

    cls: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        """Normalized area in ``[0, 1]``."""
        return self.width * self.height

    def is_valid(self) -> bool:
        """True if the box is geometrically sane and within the image."""
        if self.cls not in CLASS_NAMES:
            return False
        if not (self.width > 0 and self.height > 0):
            return False
        # Center +/- half-extent must stay inside [0, 1] (small tolerance).
        eps = 1e-6
        left = self.x_center - self.width / 2
        right = self.x_center + self.width / 2
        top = self.y_center - self.height / 2
        bottom = self.y_center + self.height / 2
        return -eps <= left and right <= 1 + eps and -eps <= top and bottom <= 1 + eps


# Issue severities returned by :func:`parse_label_file`.
SEVERITY_ERROR = "error"        # structural: cannot be trusted -> should block training
SEVERITY_FIXABLE = "fixable"    # geometric: auto-correctable (clip/drop) -> warn only


@dataclass(frozen=True)
class LabelIssue:
    """A single problem found while parsing a label file."""

    severity: str
    line: int
    message: str

    def __str__(self) -> str:  # keeps log lines readable
        return f"line {self.line} [{self.severity}]: {self.message}"


def parse_label_file(path: str | Path) -> tuple[list[YoloBox], list[LabelIssue]]:
    """Parse a YOLO label file.

    Returns ``(boxes, issues)``. Every successfully *parsed* row becomes a
    :class:`YoloBox` (even if its geometry is slightly off, so downstream
    clipping/dropping can decide what to do). ``issues`` classifies problems by
    severity:

        * :data:`SEVERITY_ERROR`  — structural (wrong field count, non-numeric,
          unknown class id). These indicate a broken/mismatched dataset and
          should block training.
        * :data:`SEVERITY_FIXABLE` — geometric (coordinates out of ``[0, 1]``,
          zero/negative size). These occur in real D-Fire labels and are
          auto-corrected by the trainer (Ultralytics clips out-of-bounds boxes
          and ignores degenerate ones), so they are warnings, not failures.

    A missing or empty file is valid (a negative image) and yields ``([], [])``.
    """
    p = Path(path)
    boxes: list[YoloBox] = []
    issues: list[LabelIssue] = []

    if not p.is_file():
        return boxes, issues  # missing label file == negative image

    with p.open("r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    for lineno, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(
                LabelIssue(SEVERITY_ERROR, lineno, f"expected 5 fields, got {len(parts)}")
            )
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            issues.append(LabelIssue(SEVERITY_ERROR, lineno, f"non-numeric value ({line!r})"))
            continue

        if cls not in CLASS_NAMES:
            issues.append(LabelIssue(SEVERITY_ERROR, lineno, f"unknown class id {cls}"))
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            issues.append(
                LabelIssue(SEVERITY_FIXABLE, lineno, f"center out of [0,1] (xc={xc}, yc={yc})")
            )
        if w <= 0.0 or h <= 0.0:
            msg = f"non-positive size (w={w}, h={h}) -> will be dropped"
            issues.append(LabelIssue(SEVERITY_FIXABLE, lineno, msg))
        elif w > 1.0 or h > 1.0:
            msg = f"size exceeds image (w={w}, h={h}) -> will be clipped"
            issues.append(LabelIssue(SEVERITY_FIXABLE, lineno, msg))

        boxes.append(YoloBox(cls, xc, yc, w, h))

    return boxes, issues


def label_path_for_image(image_path: str | Path) -> Path:
    """Map ``.../images/foo.jpg`` to ``.../labels/foo.txt``.

    Mirrors Ultralytics' convention of swapping the ``images`` path component
    for ``labels`` and replacing the suffix with ``.txt``.
    """
    p = Path(image_path)
    parts = list(p.parts)
    # Replace the last occurrence of an "images" component with "labels".
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def iter_split(images_dir: str | Path) -> Iterator[tuple[Path, Path]]:
    """Yield ``(image_path, label_path)`` pairs for every image in ``images_dir``.

    The label path is returned even if the file does not exist (negative image),
    so callers can decide how to treat missing labels.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() in IMAGE_EXTENSIONS:
            yield img, label_path_for_image(img)


def image_category(boxes: list[YoloBox]) -> str:
    """Collapse a set of boxes into one of :data:`CATEGORIES`."""
    has_smoke = any(b.cls == SMOKE_ID for b in boxes)
    has_fire = any(b.cls == FIRE_ID for b in boxes)
    if has_fire and has_smoke:
        return CATEGORY_BOTH
    if has_fire:
        return CATEGORY_FIRE
    if has_smoke:
        return CATEGORY_SMOKE
    return CATEGORY_NONE


def yolo_to_pixel(
    box: YoloBox, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Convert a normalized YOLO box to pixel ``(x1, y1, x2, y2)`` corners."""
    x1 = (box.x_center - box.width / 2) * img_w
    y1 = (box.y_center - box.height / 2) * img_h
    x2 = (box.x_center + box.width / 2) * img_w
    y2 = (box.y_center + box.height / 2) * img_h
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
