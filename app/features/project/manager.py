"""Owns the currently open Project and exposes it to the UI via signals.

Every mutation autosaves immediately — there is no explicit Save step and
no "unsaved changes" state to track.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from app.features.project.models import TASK_CLASSIFICATION, Annotation, ClassLabel, ImageEntry, Point, Project
from app.shared.class_palette import default_shortcut, palette_color
from app.shared.paths import IMAGE_EXTENSIONS, safe_folder_name

MODE_COPY = "copy"
MODE_MOVE = "move"


class ProjectController(QObject):
    """Mediates all mutation of the active Project and notifies listeners."""

    project_opened = Signal()
    project_closed = Signal()

    images_added = Signal(list)  # list[ImageEntry]
    images_removed = Signal(list)  # list[image_id] — untracked, not deleted from disk
    image_updated = Signal(str)  # image_id

    classes_changed = Signal()
    active_class_changed = Signal(str)  # class_id or ""

    def __init__(self) -> None:
        super().__init__()
        self._project: Project | None = None
        self._batch_depth = 0
        self._batch_dirty = False
        self._batch_updated_images: set[str] = set()

    # -- lifecycle -----------------------------------------------------

    @property
    def project(self) -> Project | None:
        return self._project

    @property
    def is_open(self) -> bool:
        return self._project is not None

    def set_project(self, project: Project) -> None:
        self._project = project
        self.save()
        self.project_opened.emit()

    def open_project(self, project_dir: Path) -> None:
        self.set_project(Project.load(project_dir))

    def close_project(self) -> None:
        self._project = None
        self.project_closed.emit()

    def save(self) -> None:
        if self._batch_depth > 0:
            self._batch_dirty = True
            return
        if self._project is not None:
            self._project.save()

    def begin_batch(self) -> None:
        """Defers every save() until end_batch() drops the depth back to
        zero, which then writes at most once. Also coalesces image_updated:
        listeners like the class-toggle panel and the progress widget
        recompute their *entire* state from scratch on every single
        image_updated (e.g. rebuilding every class button, each with a
        fresh per-class annotation count scanned over every image) — fine
        for one manual edit, but detection/segmentation auto-labeling fires
        image_updated once per predicted box, not per image, so a handful
        of images can mean hundreds of full panel rebuilds in a row. Callers
        that mutate the project many times in a tight loop (auto-labeling)
        must wrap the whole loop in this (see AutoLabelRunDialog, the only
        current caller)."""
        self._batch_depth += 1

    def end_batch(self) -> None:
        self._batch_depth -= 1
        if self._batch_depth > 0:
            return
        if self._batch_dirty:
            self._batch_dirty = False
            self.save()
        updated, self._batch_updated_images = self._batch_updated_images, set()
        for image_id in updated:
            self.image_updated.emit(image_id)

    # -- images ----------------------------------------------------------

    def import_paths(self, paths: list[Path]) -> int:
        project = self._require_project()
        existing = {str(Path(i.path).resolve()) for i in project.images}
        added: list[ImageEntry] = []
        for path in paths:
            resolved = str(path.resolve())
            if resolved in existing or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            entry = ImageEntry(path=resolved)
            project.images.append(entry)
            existing.add(resolved)
            added.append(entry)
        if added:
            self.save()
            self.images_added.emit(added)
        return len(added)

    def import_folder(self, folder: Path) -> int:
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        return self.import_paths(paths)

    def remove_images(self, image_ids: list[str]) -> int:
        """Untracks images from the project. Never touches the files on disk —
        this only removes them (and any annotations) from project.json."""
        project = self._require_project()
        wanted = set(image_ids)
        before = len(project.images)
        project.images = [i for i in project.images if i.id not in wanted]
        removed = before - len(project.images)
        if removed:
            self.save()
            self.images_removed.emit(image_ids)
        return removed

    # -- annotations (detection / segmentation) -----------------------------

    def add_annotation(self, image_id: str, class_id: str, points: list[Point]) -> Annotation:
        """Adds a bounding box (2 points) or polygon (3+ points) to an image."""
        project = self._require_project()
        entry = project.find_image(image_id)
        if entry is None:
            raise ValueError(f"Unknown image: {image_id}")
        annotation = Annotation(class_id=class_id, points=points)
        entry.annotations.append(annotation)
        self.save()
        self._notify_image_updated(image_id)
        return annotation

    def update_annotation(
        self,
        image_id: str,
        annotation_id: str,
        points: list[Point] | None = None,
        class_id: str | None = None,
    ) -> None:
        project = self._require_project()
        entry = project.find_image(image_id)
        annotation = next((a for a in entry.annotations if a.id == annotation_id), None) if entry else None
        if annotation is None:
            return
        if points is not None:
            annotation.points = points
        if class_id is not None:
            annotation.class_id = class_id
        self.save()
        self._notify_image_updated(image_id)

    def remove_annotation(self, image_id: str, annotation_id: str) -> None:
        project = self._require_project()
        entry = project.find_image(image_id)
        if entry is None:
            return
        before = len(entry.annotations)
        entry.annotations = [a for a in entry.annotations if a.id != annotation_id]
        if len(entry.annotations) != before:
            self.save()
            self._notify_image_updated(image_id)

    def assign_images(self, image_ids: list[str], class_id: str, mode: str) -> list[tuple[str, str]]:
        """Copies or moves each image's file into the class's folder and tags it.

        Returns a list of (image_id, error_message) for any images that failed;
        the rest still succeed.
        """
        project = self._require_project()
        label = project.find_class(class_id)
        if label is None:
            return []

        dest_dir = project.project_dir / safe_folder_name(label.name)
        dest_dir.mkdir(parents=True, exist_ok=True)

        failures: list[tuple[str, str]] = []
        changed = False
        for image_id in image_ids:
            entry = project.find_image(image_id)
            if entry is None:
                continue
            source = Path(entry.path)
            target = dest_dir / source.name
            if target.exists() and target.resolve() != source.resolve():
                target = dest_dir / f"{source.stem}_{entry.id}{source.suffix}"
            try:
                if source.resolve() != target.resolve():
                    if mode == MODE_MOVE:
                        shutil.move(str(source), str(target))
                    else:
                        shutil.copy2(source, target)
                entry.path = str(target)
                entry.class_id = class_id
                changed = True
                self._notify_image_updated(image_id)
            except OSError as exc:
                failures.append((image_id, str(exc)))

        if changed:
            self.save()
        return failures

    # -- classes -----------------------------------------------------------

    def add_class(self, name: str, color: str, shortcut: str = "") -> ClassLabel:
        project = self._require_project()
        label = ClassLabel(name=name, color=color, shortcut=shortcut)
        project.classes.append(label)
        if project.task_type == TASK_CLASSIFICATION:
            (project.project_dir / safe_folder_name(label.name)).mkdir(parents=True, exist_ok=True)
        if project.active_class_id is None:
            project.active_class_id = label.id
            self.active_class_changed.emit(label.id)
        self.save()
        self.classes_changed.emit()
        return label

    def import_classes(self, names: list[str]) -> list[ClassLabel]:
        """Bulk-creates classes (and their folders) from a class-names file."""
        project = self._require_project()
        existing_names = {c.name.lower() for c in project.classes}
        added: list[ClassLabel] = []
        for raw_name in names:
            name = raw_name.strip()
            if not name or name.lower() in existing_names:
                continue
            index = len(project.classes)
            label = ClassLabel(name=name, color=palette_color(index), shortcut=default_shortcut(index))
            project.classes.append(label)
            if project.task_type == TASK_CLASSIFICATION:
                (project.project_dir / safe_folder_name(label.name)).mkdir(parents=True, exist_ok=True)
            existing_names.add(name.lower())
            added.append(label)
        if added:
            if project.active_class_id is None:
                project.active_class_id = project.classes[0].id
                self.active_class_changed.emit(project.active_class_id)
            self.save()
            self.classes_changed.emit()
        return added

    def update_class(self, class_id: str, name: str, color: str, shortcut: str) -> None:
        project = self._require_project()
        label = project.find_class(class_id)
        if label is None:
            return

        if name != label.name and project.task_type == TASK_CLASSIFICATION:
            old_dir = project.project_dir / safe_folder_name(label.name)
            new_dir = project.project_dir / safe_folder_name(name)
            if old_dir.exists() and not new_dir.exists():
                old_dir.rename(new_dir)
                for image in project.images:
                    if image.class_id == class_id and Path(image.path).parent == old_dir:
                        image.path = str(new_dir / Path(image.path).name)
                        self._notify_image_updated(image.id)
            else:
                new_dir.mkdir(parents=True, exist_ok=True)

        label.name, label.color, label.shortcut = name, color, shortcut
        self.save()
        self.classes_changed.emit()

    def remove_class(self, class_id: str) -> None:
        project = self._require_project()
        project.classes = [c for c in project.classes if c.id != class_id]
        for image in project.images:
            if image.class_id == class_id:
                image.class_id = None
            if image.annotations:
                image.annotations = [a for a in image.annotations if a.class_id != class_id]
        if project.active_class_id == class_id:
            project.active_class_id = project.classes[0].id if project.classes else None
            self.active_class_changed.emit(project.active_class_id or "")
        self.save()
        self.classes_changed.emit()

    def set_active_class(self, class_id: str | None) -> None:
        project = self._require_project()
        if project.active_class_id == class_id:
            return
        project.active_class_id = class_id
        self.save()
        self.active_class_changed.emit(class_id or "")

    def _notify_image_updated(self, image_id: str) -> None:
        if self._batch_depth > 0:
            self._batch_updated_images.add(image_id)
            return
        self.image_updated.emit(image_id)

    def _require_project(self) -> Project:
        if self._project is None:
            raise RuntimeError("No project is open")
        return self._project
