"""Exclusive toggle buttons: exactly one class is 'active' at a time.

Turning a class's toggle on makes it the active class — images assigned
via the workspace's Assign / Move actions go to whichever class is on.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.features.project.manager import ProjectController
from app.features.project.models import TASK_CLASSIFICATION
from app.shared.class_palette import palette_color
from app.shared.labeling.class_dialog import ClassDialog
from app.shared.theme import BG_MAIN, BG_SURFACE, TEXT_PRIMARY

ClassIdPropertyKey = b"class_id"


class ClassTogglePanel(QWidget):
    active_class_changed = Signal(str)

    def __init__(self, controller: ProjectController) -> None:
        super().__init__()
        self._controller = controller
        self._refresh_pending = False
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        add_button = QPushButton("+ Add Class")
        add_button.clicked.connect(self._add_class)

        self._button_container = QWidget()
        self._button_layout = QVBoxLayout(self._button_container)
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._button_container)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(add_button)
        layout.addWidget(scroll)

        controller.project_opened.connect(self.refresh)
        controller.classes_changed.connect(self.refresh)
        controller.images_added.connect(lambda _added: self._schedule_refresh())
        controller.images_removed.connect(lambda _removed: self._schedule_refresh())
        controller.image_updated.connect(lambda _id: self._schedule_refresh())
        controller.active_class_changed.connect(self._sync_checked)
        # The workspace (and this panel) is constructed *inside* the
        # project_opened handler, i.e. always after that signal already
        # fired — so an initial refresh() can't be relied on to come from
        # the signal alone. Without this, opening an existing project that
        # already has classes would show an empty panel until some later
        # edit happened to trigger a refresh.
        self.refresh()

    def _schedule_refresh(self) -> None:
        """Coalesces a burst of image_updated/images_added/images_removed
        signals (e.g. hundreds of predictions applied back-to-back during
        auto-labeling) into a single refresh() — this panel rebuilds every
        class button from scratch, each with its own annotation count
        scanned over every image, so doing that once per prediction instead
        of once per burst was the dominant cost in a real auto-label run."""
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self._run_scheduled_refresh)

    def _run_scheduled_refresh(self) -> None:
        self._refresh_pending = False
        self.refresh()

    def refresh(self) -> None:
        for button in self._button_group.buttons():
            self._button_group.removeButton(button)
        while self._button_layout.count() > 1:
            item = self._button_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Detach immediately so the old button stops painting;
                # deleteLater() alone leaves it parented (and visible)
                # until the event loop gets around to destroying it.
                widget.setParent(None)
                widget.deleteLater()

        project = self._controller.project
        if project is None:
            return

        count_fn = project.class_image_count if project.task_type == TASK_CLASSIFICATION else project.class_annotation_count
        for label in project.classes:
            count = count_fn(label.id)
            button = QPushButton(f"{label.name}  ({count})")
            button.setCheckable(True)
            button.setChecked(label.id == project.active_class_id)
            button.setProperty("class_id", label.id)
            button.setCursor(Qt.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setMinimumHeight(32)
            button.setToolTip(f"Shortcut: {label.shortcut}" if label.shortcut else "Right-click to edit or remove")
            button.setStyleSheet(self._button_style(label.color))
            button.toggled.connect(lambda checked, class_id=label.id: self._on_toggled(class_id, checked))
            button.setContextMenuPolicy(Qt.CustomContextMenu)
            button.customContextMenuRequested.connect(
                lambda pos, b=button, class_id=label.id: self._show_context_menu(b, class_id, pos)
            )
            self._button_group.addButton(button)
            self._button_layout.insertWidget(self._button_layout.count() - 1, button)

    @staticmethod
    def _button_style(color: str) -> str:
        return f"""
            QPushButton {{
                border: 2px solid {color};
                border-radius: 8px;
                padding: 8px 12px;
                text-align: left;
                background-color: {BG_SURFACE};
                color: {TEXT_PRIMARY};
            }}
            QPushButton:hover {{
                background-color: {BG_MAIN};
            }}
            QPushButton:checked {{
                background-color: {color};
                color: white;
                font-weight: 600;
            }}
        """

    def _sync_checked(self, class_id: str) -> None:
        for button in self._button_group.buttons():
            if button.property("class_id") == class_id and not button.isChecked():
                button.setChecked(True)

    def _on_toggled(self, class_id: str, checked: bool) -> None:
        if checked:
            self._controller.set_active_class(class_id)

    def _add_class(self) -> None:
        project = self._controller.project
        if project is None:
            return
        suggested_color = palette_color(len(project.classes))
        dialog = ClassDialog(color=suggested_color, parent=self)
        if dialog.exec():
            name, color, shortcut = dialog.values()
            if name:
                self._controller.add_class(name, color, shortcut)

    def _show_context_menu(self, button: QPushButton, class_id: str, pos) -> None:
        menu = QMenu(self)
        edit_action = menu.addAction("Edit…")
        remove_action = menu.addAction("Remove")
        chosen = menu.exec(button.mapToGlobal(pos))
        if chosen == edit_action:
            self._edit_class(class_id)
        elif chosen == remove_action:
            self._remove_class(class_id)

    def _remove_class(self, class_id: str) -> None:
        project = self._controller.project
        label = project.find_class(class_id) if project else None
        if project is None or label is None:
            return
        if project.task_type == TASK_CLASSIFICATION:
            count = project.class_image_count(class_id)
            unit = "image" if count == 1 else "images"
        else:
            count = project.class_annotation_count(class_id)
            unit = "object" if count == 1 else "objects"
        detail = f" It is currently used by {count} {unit}." if count else ""
        reply = QMessageBox.question(
            self,
            "Remove Class",
            f"Remove '{label.name}'? This can't be undone.{detail}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._controller.remove_class(class_id)

    def _edit_class(self, class_id: str) -> None:
        project = self._controller.project
        label = project.find_class(class_id) if project else None
        if label is None:
            return
        dialog = ClassDialog(label.name, label.color, label.shortcut, parent=self)
        if dialog.exec():
            name, color, shortcut = dialog.values()
            if name:
                self._controller.update_class(class_id, name, color, shortcut)
