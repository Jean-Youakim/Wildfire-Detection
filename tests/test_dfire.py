"""Unit tests for the D-Fire core library (label parsing, categories, geometry)."""

from __future__ import annotations

from pathlib import Path

from src.data.dfire import (
    CATEGORY_BOTH,
    CATEGORY_FIRE,
    CATEGORY_NONE,
    CATEGORY_SMOKE,
    FIRE_ID,
    SEVERITY_ERROR,
    SEVERITY_FIXABLE,
    SMOKE_ID,
    YoloBox,
    image_category,
    label_path_for_image,
    parse_label_file,
    yolo_to_pixel,
)


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "label.txt"
    p.write_text(content, encoding="utf-8")
    return p


class TestParseLabelFile:
    def test_missing_file_is_negative(self, tmp_path):
        boxes, issues = parse_label_file(tmp_path / "does_not_exist.txt")
        assert boxes == [] and issues == []

    def test_empty_file_is_negative(self, tmp_path):
        boxes, issues = parse_label_file(_write(tmp_path, ""))
        assert boxes == [] and issues == []

    def test_valid_rows(self, tmp_path):
        p = _write(tmp_path, "0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n")
        boxes, issues = parse_label_file(p)
        assert len(boxes) == 2 and not issues
        assert boxes[0].cls == SMOKE_ID and boxes[1].cls == FIRE_ID

    def test_wrong_field_count_is_error(self, tmp_path):
        boxes, issues = parse_label_file(_write(tmp_path, "0 0.5 0.5 0.2\n"))
        assert not boxes
        assert issues and issues[0].severity == SEVERITY_ERROR

    def test_non_numeric_is_error(self, tmp_path):
        _, issues = parse_label_file(_write(tmp_path, "0 a b c d\n"))
        assert issues and issues[0].severity == SEVERITY_ERROR

    def test_unknown_class_is_error(self, tmp_path):
        _, issues = parse_label_file(_write(tmp_path, "7 0.5 0.5 0.2 0.2\n"))
        assert any(i.severity == SEVERITY_ERROR for i in issues)

    def test_out_of_range_center_is_fixable(self, tmp_path):
        boxes, issues = parse_label_file(_write(tmp_path, "0 1.5 0.5 0.2 0.2\n"))
        assert len(boxes) == 1  # still parsed so the trainer can clip it
        assert any(i.severity == SEVERITY_FIXABLE for i in issues)

    def test_zero_size_is_fixable(self, tmp_path):
        _, issues = parse_label_file(_write(tmp_path, "1 0.5 0.5 0.0 0.1\n"))
        assert any(i.severity == SEVERITY_FIXABLE for i in issues)

    def test_blank_lines_skipped(self, tmp_path):
        boxes, issues = parse_label_file(_write(tmp_path, "\n0 0.5 0.5 0.2 0.2\n\n"))
        assert len(boxes) == 1 and not issues


class TestYoloBox:
    def test_valid_box(self):
        assert YoloBox(SMOKE_ID, 0.5, 0.5, 0.2, 0.2).is_valid()

    def test_box_outside_image_invalid(self):
        assert not YoloBox(SMOKE_ID, 0.95, 0.5, 0.2, 0.2).is_valid()

    def test_zero_area_invalid(self):
        assert not YoloBox(FIRE_ID, 0.5, 0.5, 0.0, 0.2).is_valid()

    def test_unknown_class_invalid(self):
        assert not YoloBox(9, 0.5, 0.5, 0.2, 0.2).is_valid()

    def test_area(self):
        assert YoloBox(FIRE_ID, 0.5, 0.5, 0.5, 0.4).area == 0.2


class TestImageCategory:
    def test_none(self):
        assert image_category([]) == CATEGORY_NONE

    def test_smoke_only(self):
        assert image_category([YoloBox(SMOKE_ID, 0.5, 0.5, 0.1, 0.1)]) == CATEGORY_SMOKE

    def test_fire_only(self):
        assert image_category([YoloBox(FIRE_ID, 0.5, 0.5, 0.1, 0.1)]) == CATEGORY_FIRE

    def test_both(self):
        boxes = [YoloBox(SMOKE_ID, 0.5, 0.5, 0.1, 0.1), YoloBox(FIRE_ID, 0.3, 0.3, 0.1, 0.1)]
        assert image_category(boxes) == CATEGORY_BOTH


class TestGeometry:
    def test_yolo_to_pixel_roundtrip(self):
        box = YoloBox(FIRE_ID, 0.5, 0.5, 0.5, 0.5)
        assert yolo_to_pixel(box, 100, 200) == (25, 50, 75, 150)

    def test_label_path_for_image(self):
        img = Path("data") / "train" / "images" / "foo.jpg"
        assert label_path_for_image(img) == Path("data") / "train" / "labels" / "foo.txt"
