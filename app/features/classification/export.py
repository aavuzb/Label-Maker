"""Dataset export for classification projects — ImageFolder and CSV manifest formats.

Only images that have been assigned a class are exported (an unlabeled image
isn't useful to a classifier's training set). Images are split into
train/val/test by shuffling with a fixed seed, so re-exporting the same
project with the same ratios reproduces the same split.

Unlike detection/segmentation, images here already live sorted into
per-class folders inside the project directory (Assign copies/moves them
there immediately — see the workspace's module docstring) — this export
still copies them again into a *separate* output location, since a
train/val/test split and a flat "one folder per class" layout aren't what
the project directory itself is organized as.
"""

from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from app.features.project.models import ImageEntry, Project
from app.shared.paths import safe_folder_name

FORMAT_IMAGEFOLDER = "imagefolder"
FORMAT_CSV = "csv"

FORMAT_LABELS = {
    FORMAT_IMAGEFOLDER: "ImageFolder (folder per class)",
    FORMAT_CSV: "CSV manifest",
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
    skipped_unlabeled: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)


def _labeled(entry: ImageEntry) -> bool:
    return entry.class_id is not None


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
    labeled = [e for e in project.images if _labeled(e)]
    entries_by_split = split_entries(labeled, train_ratio, val_ratio, test_ratio)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == FORMAT_IMAGEFOLDER:
        result = _export_imagefolder(project, entries_by_split, out_dir)
    elif fmt == FORMAT_CSV:
        result = _export_csv(project, entries_by_split, out_dir)
    else:
        raise ValueError(f"Unknown export format: {fmt}")

    result.skipped_unlabeled = len(project.images) - len(labeled)
    return result


# -- ImageFolder -------------------------------------------------------------
#
#   out_dir/
#     {train,val,test}/{class_name}/*.jpg   — the layout torchvision's
#                                              ImageFolder, Keras'
#                                              flow_from_directory, and most
#                                              other classifier training
#                                              tools expect directly.


def _export_imagefolder(project: Project, entries_by_split: dict[str, list[ImageEntry]], out_dir: Path) -> ExportResult:
    result = ExportResult(FORMAT_IMAGEFOLDER, out_dir)

    for split, entries in entries_by_split.items():
        count = 0
        for entry in entries:
            source = Path(entry.path)
            if not source.exists():
                result.skipped_unreadable += 1
                continue
            label = project.find_class(entry.class_id)
            if label is None:
                continue

            class_dir = out_dir / split / safe_folder_name(label.name)
            class_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, class_dir / _dest_name(entry, class_dir))

            count += 1
            result.exported_count += 1
        result.split_counts[split] = count

    return result


# -- CSV manifest -------------------------------------------------------------
#
#   out_dir/
#     images/{train,val,test}/*.jpg
#     {train,val,test}.csv   (columns: filename, label)


def _export_csv(project: Project, entries_by_split: dict[str, list[ImageEntry]], out_dir: Path) -> ExportResult:
    result = ExportResult(FORMAT_CSV, out_dir)

    for split, entries in entries_by_split.items():
        image_dir = out_dir / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)

        rows: list[tuple[str, str]] = []
        for entry in entries:
            source = Path(entry.path)
            if not source.exists():
                result.skipped_unreadable += 1
                continue
            label = project.find_class(entry.class_id)
            if label is None:
                continue

            dest_name = _dest_name(entry, image_dir)
            shutil.copy2(source, image_dir / dest_name)
            rows.append((dest_name, label.name))

            result.exported_count += 1

        with (out_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(rows)
        result.split_counts[split] = len(rows)

    return result
