"""New Project and Open Project dialogs."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.features.project.models import (
    TASK_CLASSIFICATION,
    TASK_DETECTION,
    TASK_SEGMENTATION,
    Project,
)
from app.shared.paths import PROJECTS_DIR, ensure_app_dirs
from app.shared.theme import (
    ACCENT,
    ACCENT_TEXT,
    FS_BODY,
    FS_CAPTION,
    FS_H1,
    TASK_ACCENTS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    primary_button_style,
    tint_rgba,
)

_TASK_LABELS = {
    TASK_CLASSIFICATION: "Classification",
    TASK_DETECTION: "Detection",
    TASK_SEGMENTATION: "Segmentation",
}

_SLUG_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]+")

_ERROR_COLOR = "#dc2626"


def _slugify(name: str) -> str:
    slug = _SLUG_UNSAFE.sub("-", name.strip()).strip("-").lower()
    return slug or "project"


def _read_project_name(project_dir: Path) -> str | None:
    try:
        return Project.load(project_dir).name
    except Exception:  # noqa: BLE001 — best-effort, just for a friendlier warning message
        return None


class NewProjectDialog(QDialog):
    """Asks for a project name and builds a Project for the given task type.

    A name that collides with an existing project's folder isn't silently
    suffixed ("project1-2", "project1-3", …) — project names must stay
    unique and meaningful, so this warns and lets the user either pick a
    different name or replace the existing project outright.
    """

    def __init__(self, parent: QWidget, task_type: str) -> None:
        super().__init__(parent)
        self._task_type = task_type
        self._project: Project | None = None
        task_label = _TASK_LABELS.get(task_type, task_type)
        accent = TASK_ACCENTS.get(task_type, TASK_ACCENTS[TASK_CLASSIFICATION])

        self.setWindowTitle(f"New {task_label} Project")
        self.setMinimumSize(520, 260)
        self.resize(560, 280)

        heading = QLabel(f"New {task_label} Project")
        heading.setStyleSheet(f"font-size: {FS_H1}px; font-weight: 700; color: {TEXT_PRIMARY}; border: none;")

        hint = QLabel("Choose a name for this project — it names the folder that will hold its images, classes, and labels.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FS_BODY}px; border: none;")

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. plant-diseases")
        self.name_edit.setMinimumHeight(32)
        self.name_edit.returnPressed.connect(self._on_create_clicked)
        self.name_edit.textEdited.connect(lambda _text: self._clear_error())

        form = QFormLayout()
        form.setSpacing(10)
        form.addRow("Project name", self.name_edit)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"color: {_ERROR_COLOR}; font-size: {FS_CAPTION}px; border: none;")
        self.error_label.setVisible(False)

        self.create_button = QPushButton("Create Project")
        self.create_button.setStyleSheet(primary_button_style(accent))
        self.create_button.setDefault(True)
        self.create_button.clicked.connect(self._on_create_clicked)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.addButton(self.create_button, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)
        layout.addWidget(heading)
        layout.addWidget(hint)
        layout.addSpacing(4)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addStretch(1)
        layout.addWidget(buttons)

        self.name_edit.setFocus()

    def _clear_error(self) -> None:
        self.error_label.setVisible(False)

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(True)

    def _on_create_clicked(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            self._show_error("Enter a project name.")
            return
        self._clear_error()

        ensure_app_dirs()
        candidate = PROJECTS_DIR / _slugify(name)
        if candidate.exists():
            if not self._resolve_duplicate(candidate):
                return  # user chose to pick a different name (or dismissed the warning)
            try:
                shutil.rmtree(candidate)
            except OSError as exc:
                self._show_error(f"Couldn't remove the existing project: {exc}")
                return

        self._project = Project(name=name, task_type=self._task_type, project_dir=candidate)
        self.accept()

    def _resolve_duplicate(self, candidate: Path) -> bool:
        """Returns True if the caller should delete `candidate` and proceed
        (the user chose Replace); False if creation should be aborted so the
        user can pick a different name (the default, safer outcome — also
        what happens if the warning is dismissed without an explicit choice)."""
        existing_name = _read_project_name(candidate)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Project Already Exists")
        if existing_name:
            box.setText(f"A project named “{existing_name}” already exists.")
        else:
            box.setText(f"A folder named “{candidate.name}” already exists.")
        box.setInformativeText(
            "Project names must be unique. Choose a different name, or replace the existing project — "
            "this permanently deletes it."
        )
        replace_button = box.addButton("Replace Existing Project", QMessageBox.DestructiveRole)
        rename_button = box.addButton("Choose a Different Name", QMessageBox.RejectRole)
        box.setDefaultButton(rename_button)
        box.exec()

        if box.clickedButton() is replace_button:
            return True
        self.name_edit.setFocus()
        self.name_edit.selectAll()
        return False

    def project(self) -> Project | None:
        return self._project


def prompt_new_project(parent: QWidget, task_type: str) -> Project | None:
    """Asks for a project name and builds a Project for the given task type."""
    dialog = NewProjectDialog(parent, task_type)
    if dialog.exec():
        return dialog.project()
    return None


def _task_dot_icon(accent: str, size: int = 14) -> QIcon:
    """A small filled circle in a task's accent color, used as a list-item
    icon so each row's task type reads at a glance — the same color already
    used for that task's cards/buttons/progress bar elsewhere in the app."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(accent))
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    return QIcon(pixmap)


class OpenProjectDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open Project")
        self.setMinimumSize(460, 360)
        self._selected_dir: Path | None = None

        heading = QLabel("Open Project")
        heading.setStyleSheet(f"font-size: {FS_H1}px; font-weight: 700; color: {TEXT_PRIMARY}; border: none;")
        subtitle = QLabel("Choose a project to continue working on it.")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FS_BODY}px; border: none;")

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(12, 12))
        self.list_widget.setSpacing(2)
        self.list_widget.setStyleSheet(
            f"""
            QListWidget::item {{ padding: 8px 10px; border-radius: 6px; }}
            QListWidget::item:selected {{ background-color: {tint_rgba(ACCENT, 0.15)}; color: {ACCENT_TEXT}; }}
            QListWidget::item:hover {{ background-color: {tint_rgba(ACCENT, 0.07)}; }}
            """
        )
        self.list_widget.itemDoubleClicked.connect(self.accept)

        browse_button = QPushButton("Browse for project.json…")
        browse_button.clicked.connect(self._browse)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setStyleSheet(primary_button_style())
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(10)
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addSpacing(4)
        layout.addWidget(self.list_widget, stretch=1)
        layout.addWidget(browse_button)
        layout.addWidget(buttons)

        self._populate()

    def _populate(self) -> None:
        ensure_app_dirs()
        loaded: list[tuple[Project, Path]] = []
        for entry in sorted(PROJECTS_DIR.iterdir()):
            project_file = entry / "project.json"
            if entry.is_dir() and project_file.exists():
                try:
                    project = Project.load(entry)
                except Exception:
                    continue
                loaded.append((project, entry))

        # Two projects can legitimately share a display name (NewProjectDialog
        # only enforces uniqueness on the *folder* it's about to create, and
        # nothing stops that folder ending up with a different name later, or
        # existing projects predating that check) — the folder name is the
        # only thing that's actually guaranteed unique, so surface it
        # whenever a name collides, to tell otherwise-identical-looking rows
        # apart. Left off entries whose name is already unique, to keep the
        # common case uncluttered.
        name_counts: dict[str, int] = {}
        for project, _entry in loaded:
            name_counts[project.name] = name_counts.get(project.name, 0) + 1

        for project, entry in loaded:
            task_label = _TASK_LABELS.get(project.task_type, project.task_type)
            label = f"{project.name}  —  {task_label}  ({len(project.images)} images)"
            if name_counts[project.name] > 1:
                label += f"  [{entry.name}]"
            item = QListWidgetItem(_task_dot_icon(TASK_ACCENTS.get(project.task_type, ACCENT)), label)
            item.setData(1000, entry)
            self.list_widget.addItem(item)

    def _browse(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open project.json", str(PROJECTS_DIR), "LabelMaker Project (project.json)"
        )
        if file_path:
            self._selected_dir = Path(file_path).parent
            self.accept()

    def accept(self) -> None:
        if self._selected_dir is None:
            item = self.list_widget.currentItem()
            if item is not None:
                self._selected_dir = item.data(1000)
        if self._selected_dir is None:
            QMessageBox.information(self, "Open Project", "Select a project first.")
            return
        super().accept()

    def selected_project_dir(self) -> Path | None:
        return self._selected_dir
