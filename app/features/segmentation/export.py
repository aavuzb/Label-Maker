"""Dataset export for segmentation projects — YOLO-seg and COCO formats.

Only images that carry at least one polygon are exported (an image with no
polygons isn't useful to a segmentation training set). Images are split into
train/val/test by shuffling with a fixed seed, so re-exporting the same
project with the same ratios reproduces the same split.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtGui import QImageReader

from app.features.project.models import Annotation, ImageEntry, Project

FORMAT_YOLO = "yolo"
FORMAT_COCO = "coco"

FORMAT_LABELS = {
    FORMAT_YOLO: "YOLO-seg (.txt)",
    FORMAT_COCO: "COCO (.json)",
}

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
_SPLIT_ORDER = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)


@dataclass
class ExportResult:
    format: str
    output_dir: Path
    exported_count: int = 0
    skipped_unreadable: int = 0
    skipped_unannotated: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)


def _polygons(entry: ImageEntry) -> list[Annotation]:
    return [a for a in entry.annotations if len(a.points) >= 3]


def _image_size(path: str) -> tuple[int, int] | None:
    size = QImageReader(path).size()
    if not size.isValid() or size.isEmpty():
        return None
    return size.width(), size.height()


def _dest_name(entry: ImageEntry, dest_dir: Path) -> str:
    """A filename unique within dest_dir, disambiguating same-named files
    that originated from different source folders."""
    source = Path(entry.path)
    if not (dest_dir / source.name).exists():
        return source.name
    return f"{source.stem}_{entry.id}{source.suffix}"


def split_entries(
    entries: list[ImageEntry],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> dict[str, list[ImageEntry]]:
    """Deterministically shuffles and splits entries by ratio.

    A split with a ratio of 0 is omitted entirely — any images that would
    have rounded into it are folded into the first nonzero split instead, so
    disabling a split never silently drops images from the export.
    """
    ordered = list(entries)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)

    n_train = round(n * train_ratio) if train_ratio > 0 else 0
    n_train = min(n_train, n)
    n_val = round(n * val_ratio) if val_ratio > 0 else 0
    n_val = min(n_val, n - n_train)

    remainder = n - n_train - n_val
    if test_ratio <= 0:
        if train_ratio > 0:
            n_train += remainder
        elif val_ratio > 0:
            n_val += remainder

    buckets = {
        SPLIT_TRAIN: ordered[:n_train],
        SPLIT_VAL: ordered[n_train : n_train + n_val],
        SPLIT_TEST: ordered[n_train + n_val :],
    }
    ratios = {SPLIT_TRAIN: train_ratio, SPLIT_VAL: val_ratio, SPLIT_TEST: test_ratio}
    return {split: buckets[split] for split in _SPLIT_ORDER if ratios[split] > 0}


def export_dataset(
    project: Project,
    out_dir: Path,
    fmt: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> ExportResult:
    annotated = [e for e in project.images if _polygons(e)]
    entries_by_split = split_entries(annotated, train_ratio, val_ratio, test_ratio)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == FORMAT_YOLO:
        result = _export_yolo(project, entries_by_split, out_dir)
    elif fmt == FORMAT_COCO:
        result = _export_coco(project, entries_by_split, out_dir)
    else:
        raise ValueError(f"Unknown export format: {fmt}")

    result.skipped_unannotated = len(project.images) - len(annotated)
    return result


# -- YOLO-seg ---------------------------------------------------------------
#
#   out_dir/
#     images/{train,val,test}/*.jpg
#     labels/{train,val,test}/*.txt   (one "class_id x1 y1 x2 y2 ... xn yn"
#                                       line per polygon, all normalized to [0, 1] —
#                                       the format Ultralytics YOLO-seg training expects)
#     classes.yaml


def _export_yolo(project: Project, entries_by_split: dict[str, list[ImageEntry]], out_dir: Path) -> ExportResult:
    class_index = {label.id: i for i, label in enumerate(project.classes)}
    result = ExportResult(FORMAT_YOLO, out_dir)

    for split, entries in entries_by_split.items():
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for entry in entries:
            source = Path(entry.path)
            if not source.exists():
                result.skipped_unreadable += 1
                continue

            dest_image = image_dir / _dest_name(entry, image_dir)
            shutil.copy2(source, dest_image)

            lines = []
            for polygon in _polygons(entry):
                if polygon.class_id not in class_index:
                    continue
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon.points)
                lines.append(f"{class_index[polygon.class_id]} {coords}")
            label_text = "\n".join(lines) + ("\n" if lines else "")
            (label_dir / f"{dest_image.stem}.txt").write_text(label_text, encoding="utf-8")

            count += 1
            result.exported_count += 1
        result.split_counts[split] = count

    yaml_lines = [f"{split}: images/{split}" for split in entries_by_split]
    yaml_lines.append(f"nc: {len(project.classes)}")
    names = ", ".join(json.dumps(label.name) for label in project.classes)
    yaml_lines.append(f"names: [{names}]")
    (out_dir / "classes.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return result


# -- COCO ---------------------------------------------------------------
#
#   out_dir/
#     {train,val,test}/*.jpg
#     annotations/instances_{train,val,test}.json   (COCO instance-segmentation
#                                                      format: images/annotations/
#                                                      categories, polygon coords
#                                                      in absolute pixels)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    """Shoelace formula. Assumes a simple (non-self-intersecting) polygon,
    which is what this app's own drawing/auto-label tools always produce."""
    total = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _export_coco(project: Project, entries_by_split: dict[str, list[ImageEntry]], out_dir: Path) -> ExportResult:
    annotations_dir = out_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    # COCO category ids conventionally start at 1, not 0.
    category_id = {label.id: i for i, label in enumerate(project.classes, start=1)}
    categories = [{"id": category_id[label.id], "name": label.name} for label in project.classes]

    result = ExportResult(FORMAT_COCO, out_dir)

    for split, entries in entries_by_split.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        images_json = []
        annotations_json = []
        next_image_id = 1
        next_annotation_id = 1
        for entry in entries:
            source = Path(entry.path)
            size = _image_size(str(source)) if source.exists() else None
            if size is None:
                result.skipped_unreadable += 1
                continue
            width, height = size

            dest_image = split_dir / _dest_name(entry, split_dir)
            shutil.copy2(source, dest_image)

            image_id = next_image_id
            next_image_id += 1
            images_json.append({"id": image_id, "file_name": dest_image.name, "width": width, "height": height})

            for polygon in _polygons(entry):
                if polygon.class_id not in category_id:
                    continue
                pixel_points = [(x * width, y * height) for x, y in polygon.points]
                xs = [p[0] for p in pixel_points]
                ys = [p[1] for p in pixel_points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                flat_points = [coord for point in pixel_points for coord in point]
                annotations_json.append(
                    {
                        "id": next_annotation_id,
                        "image_id": image_id,
                        "category_id": category_id[polygon.class_id],
                        "segmentation": [flat_points],
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "area": _polygon_area(pixel_points),
                        "iscrowd": 0,
                    }
                )
                next_annotation_id += 1

            result.exported_count += 1

        (annotations_dir / f"instances_{split}.json").write_text(
            json.dumps({"images": images_json, "annotations": annotations_json, "categories": categories}, indent=2),
            encoding="utf-8",
        )
        result.split_counts[split] = len(images_json)

    return result
