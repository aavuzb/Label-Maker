"""Project data model and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.shared.ids import new_id

TASK_CLASSIFICATION = "classification"
TASK_DETECTION = "detection"
TASK_SEGMENTATION = "segmentation"

TASK_TYPES = (TASK_CLASSIFICATION, TASK_DETECTION, TASK_SEGMENTATION)

PROJECT_FILE_NAME = "project.json"


@dataclass
class ClassLabel:
    name: str
    color: str = "#3498db"
    shortcut: str = ""
    id: str = field(default_factory=new_id)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "color": self.color, "shortcut": self.shortcut}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ClassLabel:
        return ClassLabel(
            id=data.get("id", new_id()),
            name=data.get("name", "Class"),
            color=data.get("color", "#3498db"),
            shortcut=data.get("shortcut", ""),
        )


Point = tuple[float, float]


@dataclass
class Annotation:
    """A single labeled object within an image.

    `points` are normalized to the image's [0, 1] range so they stay valid
    regardless of the image's pixel resolution (the standard YOLO convention).
    A bounding box stores its two opposite corners (2 points); a polygon
    stores each of its vertices in order (3+ points).
    """

    class_id: str
    points: list[Point]
    id: str = field(default_factory=new_id)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "class_id": self.class_id, "points": [list(p) for p in self.points]}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Annotation:
        return Annotation(
            id=data.get("id", new_id()),
            class_id=data["class_id"],
            points=[(float(p[0]), float(p[1])) for p in data.get("points", [])],
        )


@dataclass
class ImageEntry:
    path: str
    id: str = field(default_factory=new_id)
    class_id: str | None = None
    annotations: list[Annotation] = field(default_factory=list)

    @property
    def state(self) -> str:
        return "assigned" if (self.class_id is not None or self.annotations) else "unlabeled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "class_id": self.class_id,
            "annotations": [a.to_dict() for a in self.annotations],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ImageEntry:
        return ImageEntry(
            id=data.get("id", new_id()),
            path=data["path"],
            class_id=data.get("class_id"),
            annotations=[Annotation.from_dict(a) for a in data.get("annotations", [])],
        )


@dataclass
class Project:
    name: str
    task_type: str
    project_dir: Path
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    classes: list[ClassLabel] = field(default_factory=list)
    images: list[ImageEntry] = field(default_factory=list)
    active_class_id: str | None = None

    @property
    def file_path(self) -> Path:
        return self.project_dir / PROJECT_FILE_NAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "created_at": self.created_at,
            "active_class_id": self.active_class_id,
            "classes": [c.to_dict() for c in self.classes],
            "images": [i.to_dict() for i in self.images],
        }

    def save(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(project_dir: Path) -> Project:
        data = json.loads((project_dir / PROJECT_FILE_NAME).read_text(encoding="utf-8"))
        return Project(
            id=data.get("id", new_id()),
            name=data.get("name", project_dir.name),
            task_type=data.get("task_type", TASK_CLASSIFICATION),
            project_dir=project_dir,
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            classes=[ClassLabel.from_dict(c) for c in data.get("classes", [])],
            images=[ImageEntry.from_dict(i) for i in data.get("images", [])],
            active_class_id=data.get("active_class_id"),
        )

    def find_class(self, class_id: str | None) -> ClassLabel | None:
        if class_id is None:
            return None
        return next((c for c in self.classes if c.id == class_id), None)

    def find_image(self, image_id: str) -> ImageEntry | None:
        return next((i for i in self.images if i.id == image_id), None)

    def class_image_count(self, class_id: str) -> int:
        return sum(1 for i in self.images if i.class_id == class_id)

    def class_annotation_count(self, class_id: str) -> int:
        return sum(1 for i in self.images for a in i.annotations if a.class_id == class_id)
