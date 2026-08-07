"""Dataset export for detection projects — YOLO, Pascal VOC, and CreateML formats.

Only images that carry at least one bounding box are exported (an image with
no boxes isn't useful to a detector's training set). Images are split into
train/val/test by shuffling with a fixed seed, so re-exporting the same
project with the same ratios reproduces the same split.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from PySide6.QtGui import QImageReader

from app.features.project.models import Annotation, ImageEntry, Project

FORMAT_YOLO = "yolo"
FORMAT_VOC = "voc"
FORMAT_CREATEML = "createml"

FORMAT_LABELS = {
    FORMAT_YOLO: "YOLO (.txt)",
    FORMAT_VOC: "Pascal VOC (.xml)",
    FORMAT_CREATEML: "CreateML (.json)",
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


def _boxes(entry: ImageEntry) -> list[Annotation]:
    return [a for a in entry.annotations if len(a.points) == 2]


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
    annotated = [e for e in project.images if _boxes(e)]
    entries_by_split = split_entries(annotated, train_ratio, val_ratio, test_ratio)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == FORMAT_YOLO:
        result = _export_yolo(project, entries_by_split, out_dir)
    elif fmt == FORMAT_VOC:
        result = _export_voc(project, entries_by_split, out_dir)
    elif fmt == FORMAT_CREATEML:
        result = _export_createml(project, entries_by_split, out_dir)
    else:
        raise ValueError(f"Unknown export format: {fmt}")

    result.skipped_unannotated = len(project.images) - len(annotated)
    return result


# -- YOLO -----------------------------------------------------------------
#
#   out_dir/
#     images/{train,val,test}/*.jpg
#     labels/{train,val,test}/*.txt   (one "class_id cx cy w h" line per box,
#                                       all normalized to [0, 1])
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
            for box in _boxes(entry):
                if box.class_id not in class_index:
                    continue
                (x1, y1), (x2, y2) = box.points
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = abs(x2 - x1), abs(y2 - y1)
                lines.append(f"{class_index[box.class_id]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
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


# -- Pascal VOC -------------------------------------------------------------
#
#   out_dir/
#     JPEGImages/*.jpg
#     Annotations/*.xml
#     ImageSets/Main/{train,val,test}.txt   (bare filename stems, one per line)


def _export_voc(project: Project, entries_by_split: dict[str, list[ImageEntry]], out_dir: Path) -> ExportResult:
    images_dir = out_dir / "JPEGImages"
    annotations_dir = out_dir / "Annotations"
    imagesets_dir = out_dir / "ImageSets" / "Main"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    result = ExportResult(FORMAT_VOC, out_dir)

    for split, entries in entries_by_split.items():
        stems = []
        for entry in entries:
            source = Path(entry.path)
            size = _image_size(str(source)) if source.exists() else None
            if size is None:
                result.skipped_unreadable += 1
                continue
            width, height = size

            dest_image = images_dir / _dest_name(entry, images_dir)
            shutil.copy2(source, dest_image)
            _write_voc_xml(annotations_dir / f"{dest_image.stem}.xml", project, entry, dest_image.name, width, height)

            stems.append(dest_image.stem)
            result.exported_count += 1

        (imagesets_dir / f"{split}.txt").write_text("\n".join(stems) + ("\n" if stems else ""), encoding="utf-8")
        result.split_counts[split] = len(stems)

    return result


def _write_voc_xml(path: Path, project: Project, entry: ImageEntry, filename: str, width: int, height: int) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = filename
    source_el = ET.SubElement(root, "source")
    ET.SubElement(source_el, "database").text = "LabelMaker"
    size_el = ET.SubElement(root, "size")
    ET.SubElement(size_el, "width").text = str(width)
    ET.SubElement(size_el, "height").text = str(height)
    ET.SubElement(size_el, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    for box in _boxes(entry):
        label = project.find_class(box.class_id)
        if label is None:
            continue
        (x1, y1), (x2, y2) = box.points
        xmin, xmax = sorted((x1 * width, x2 * width))
        ymin, ymax = sorted((y1 * height, y2 * height))
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label.name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        box_el = ET.SubElement(obj, "bndbox")
        ET.SubElement(box_el, "xmin").text = str(round(xmin))
        ET.SubElement(box_el, "ymin").text = str(round(ymin))
        ET.SubElement(box_el, "xmax").text = str(round(xmax))
        ET.SubElement(box_el, "ymax").text = str(round(ymax))

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


# -- CreateML ---------------------------------------------------------------
#
#   out_dir/{train,val,test}/*.jpg + annotations.json
#   annotations.json: [{"image": name, "annotations": [{"label", "coordinates":
#   {x, y, width, height}}]}] — coordinates are absolute pixels, (x, y) is the
#   box CENTER (Create ML's convention, not top-left).


def _export_createml(project: Project, entries_by_split: dict[str, list[ImageEntry]], out_dir: Path) -> ExportResult:
    result = ExportResult(FORMAT_CREATEML, out_dir)

    for split, entries in entries_by_split.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for entry in entries:
            source = Path(entry.path)
            size = _image_size(str(source)) if source.exists() else None
            if size is None:
                result.skipped_unreadable += 1
                continue
            width, height = size

            dest_image = split_dir / _dest_name(entry, split_dir)
            shutil.copy2(source, dest_image)

            annotations = []
            for box in _boxes(entry):
                label = project.find_class(box.class_id)
                if label is None:
                    continue
                (x1, y1), (x2, y2) = box.points
                xmin, xmax = sorted((x1 * width, x2 * width))
                ymin, ymax = sorted((y1 * height, y2 * height))
                annotations.append(
                    {
                        "label": label.name,
                        "coordinates": {
                            "x": round((xmin + xmax) / 2, 2),
                            "y": round((ymin + ymax) / 2, 2),
                            "width": round(xmax - xmin, 2),
                            "height": round(ymax - ymin, 2),
                        },
                    }
                )
            records.append({"image": dest_image.name, "annotations": annotations})
            result.exported_count += 1

        (split_dir / "annotations.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        result.split_counts[split] = len(records)

    return result
