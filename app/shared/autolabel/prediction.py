"""Shared result type + model-output-index -> project-class-id resolution."""

from __future__ import annotations

from dataclasses import dataclass

from app.features.project.manager import ProjectController
from app.features.project.models import Project, Point


@dataclass
class Prediction:
    class_id: str
    confidence: float
    # None => the prediction covers the whole image (classification).
    # 2 points => a box (top-left, bottom-right). 3+ points => a polygon.
    # All coordinates are normalized to the image's [0, 1] range.
    points: list[Point] | None = None


def resolve_class_mapping(
    controller: ProjectController, class_names: list[str] | None
) -> tuple[int, dict[int, str]]:
    """Returns (num_classes, {model_output_index: project_class_id}).

    With no `class_names`, the model is assumed to output classes in the
    same order the project's classes are listed in — index i maps to the
    i-th class. With `class_names` (one name per line, in the model's
    output order), any name that doesn't already exist as a project class
    is created first (mirroring "Import Classes"), then matched by name.
    """
    project = controller.project
    if project is None:
        return 0, {}

    if not class_names:
        return len(project.classes), {i: label.id for i, label in enumerate(project.classes)}

    controller.import_classes(class_names)
    by_name = {label.name.strip().lower(): label.id for label in controller.project.classes}
    mapping = {}
    for index, name in enumerate(class_names):
        class_id = by_name.get(name.strip().lower())
        if class_id is not None:
            mapping[index] = class_id
    return len(class_names), mapping


def unlabeled_image_ids(project: Project) -> list[str]:
    return [entry.id for entry in project.images if entry.state == "unlabeled"]
