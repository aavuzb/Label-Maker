"""Import Folder / Import Image / Import Classes button row, shared by every
task workspace — importing is identical regardless of labeling paradigm."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QWidget

from app.features.project.manager import ProjectController


def build_import_row(controller: ProjectController, parent: QWidget) -> QHBoxLayout:
    def import_images() -> None:
        files, _ = QFileDialog.getOpenFileNames(
            parent, "Import Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff *.webp)"
        )
        if files:
            controller.import_paths([Path(f) for f in files])

    def import_folder() -> None:
        folder = QFileDialog.getExistingDirectory(parent, "Import Folder")
        if folder:
            controller.import_folder(Path(folder))

    def import_classes() -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Import Classes", "", "Class Names (*.txt *.names);;All Files (*)"
        )
        if file_path:
            controller.import_classes(Path(file_path).read_text(encoding="utf-8").splitlines())

    import_folder_button = QPushButton("Import Folder")
    import_image_button = QPushButton("Import Image")
    import_classes_button = QPushButton("Import Classes")
    import_folder_button.clicked.connect(import_folder)
    import_image_button.clicked.connect(import_images)
    import_classes_button.clicked.connect(import_classes)

    row = QHBoxLayout()
    row.setContentsMargins(12, 12, 12, 8)
    row.setSpacing(8)
    for button in (import_folder_button, import_image_button, import_classes_button):
        row.addWidget(button)
    row.addStretch()
    return row
