"""Shared "remove image(s) from project" confirmation flow.

Removing untracks images from project.json only — it never deletes the
original files on disk. Still asks for confirmation since, for detection
and segmentation, it also discards any annotations drawn on those images
and there's no undo.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget

from app.features.project.manager import ProjectController


def confirm_and_remove(parent: QWidget, controller: ProjectController, image_ids: list[str]) -> int:
    if not image_ids:
        return 0
    count = len(image_ids)
    reply = QMessageBox.question(
        parent,
        "Remove from Project",
        f"Remove {count} {'image' if count == 1 else 'images'} from this project?\n\n"
        "The original file(s) on disk are not touched — only removed from this project "
        "(along with any labels/annotations on them). This can't be undone.",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if reply != QMessageBox.Yes:
        return 0
    return controller.remove_images(image_ids)
