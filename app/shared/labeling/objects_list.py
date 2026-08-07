"""Per-image list of annotated objects.

The concrete, visible proof that one image can hold objects of many
different classes for detection/segmentation: every box/polygon drawn on
the current image shows up here with its own class, independent of
whichever class is currently "active" for drawing new shapes. From here
an object can be selected on the canvas, reassigned to a different class,
or deleted — without needing to delete and redraw it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.features.project.manager import ProjectController
from app.shared.theme import FS_CAPTION, TEXT_MUTED, TEXT_PRIMARY

AnnotationIdRole = Qt.UserRole + 1


class ObjectsListPanel(QWidget):
    object_selected = Signal(str)  # annotation_id
    delete_requested = Signal(str)  # annotation_id
    class_change_requested = Signal(str, str)  # annotation_id, new_class_id

    def __init__(self, controller: ProjectController) -> None:
        super().__init__()
        self._controller = controller
        self._image_id: str | None = None

        self.list_widget = QListWidget()
        self.list_widget.setSpacing(2)
        self.list_widget.setSelectionMode(QListWidget.NoSelection)
        self.list_widget.itemClicked.connect(self._on_item_clicked)

        self._empty_label = QLabel("Select an image, then draw a box/polygon to see it listed here.")
        self._empty_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FS_CAPTION}px; padding: 8px; border: none;")
        self._empty_label.setWordWrap(True)
        self._empty_label.setAlignment(Qt.AlignTop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.list_widget)
        layout.addWidget(self._empty_label)

        controller.image_updated.connect(self._on_image_updated)
        controller.classes_changed.connect(self.refresh)
        controller.project_closed.connect(lambda: self.set_image(None))

    def set_image(self, image_id: str | None) -> None:
        self._image_id = image_id
        self.refresh()

    def _on_image_updated(self, image_id: str) -> None:
        if image_id == self._image_id:
            self.refresh()

    def refresh(self) -> None:
        self.list_widget.clear()
        project = self._controller.project
        entry = project.find_image(self._image_id) if project and self._image_id else None
        annotations = entry.annotations if entry else []

        self._empty_label.setVisible(not annotations)
        self.list_widget.setVisible(bool(annotations))
        if not annotations:
            return

        classes = project.classes if project else []
        for annotation in annotations:
            label = project.find_class(annotation.class_id) if project else None
            color = label.color if label else "#888888"
            name = label.name if label else "Unknown class"

            item = QListWidgetItem()
            item.setData(AnnotationIdRole, annotation.id)
            self.list_widget.addItem(item)
            row = self._build_row(annotation.id, color, name, classes)
            item.setSizeHint(row.sizeHint())
            self.list_widget.setItemWidget(item, row)

    def _build_row(self, annotation_id: str, color: str, name: str, classes: list) -> QWidget:
        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(f"background-color: {color}; border-radius: 6px; border: none;")

        name_label = QLabel(name)
        name_label.setStyleSheet(f"border: none; color: {TEXT_PRIMARY};")
        name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        change_button = QPushButton("⋮")
        change_button.setFixedSize(22, 22)
        change_button.setStyleSheet("padding: 0px; font-weight: 700;")
        change_button.setToolTip("Change this object's class")
        change_button.setCursor(Qt.PointingHandCursor)
        change_button.clicked.connect(lambda: self._show_class_menu(annotation_id, change_button, classes))

        delete_button = QPushButton("×")
        delete_button.setFixedSize(22, 22)
        delete_button.setStyleSheet("padding: 0px; font-weight: 700; color: #dc2626;")
        delete_button.setToolTip("Delete this object")
        delete_button.setCursor(Qt.PointingHandCursor)
        delete_button.clicked.connect(lambda: self.delete_requested.emit(annotation_id))

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(6, 4, 6, 4)
        row_layout.setSpacing(8)
        row_layout.addWidget(swatch)
        row_layout.addWidget(name_label)
        row_layout.addWidget(change_button)
        row_layout.addWidget(delete_button)
        return row

    def _show_class_menu(self, annotation_id: str, anchor: QPushButton, classes: list) -> None:
        if not classes:
            return
        menu = QMenu(self)
        for label in classes:
            action = menu.addAction(label.name)
            action.triggered.connect(
                lambda _checked=False, cid=label.id: self.class_change_requested.emit(annotation_id, cid)
            )
        menu.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        annotation_id = item.data(AnnotationIdRole)
        if annotation_id:
            self.object_selected.emit(annotation_id)
