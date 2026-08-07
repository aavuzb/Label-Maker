"""Search-as-you-type popup for picking a class — used from the grid's
"Edit Class…" action to fix a wrong (including auto-)labeled image without
hunting through a long class list by color alone."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from app.features.project.models import ClassLabel
from app.shared.theme import BG_SURFACE, BORDER, RADIUS, TEXT_PRIMARY

ClassIdRole = Qt.UserRole + 1

_POPUP_WIDTH = 240
_LIST_MAX_HEIGHT = 220


class ClassPickerPopup(QWidget):
    class_picked = Signal(str)  # class_id

    def __init__(self, classes: list[ClassLabel], parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Popup)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setObjectName("ClassPickerPopup")
        self._classes = classes

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search classes…")
        self.search_edit.textChanged.connect(self._refresh_list)
        self.search_edit.installEventFilter(self)

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(_LIST_MAX_HEIGHT)
        self.list_widget.itemClicked.connect(self._accept_item)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.search_edit)
        layout.addWidget(self.list_widget)

        self.setFixedWidth(_POPUP_WIDTH)
        self.setStyleSheet(
            f"#ClassPickerPopup {{ background-color: {BG_SURFACE}; border: 1px solid {BORDER}; border-radius: {RADIUS}px; }}"
        )

        self._refresh_list("")

    def popup_at(self, global_pos: QPoint) -> None:
        self.adjustSize()
        self.move(self._clamp_to_screen(global_pos))
        self.show()
        self.search_edit.setFocus()

    def _clamp_to_screen(self, global_pos: QPoint) -> QPoint:
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        if screen is None:
            return global_pos
        avail = screen.availableGeometry()
        x = min(max(global_pos.x(), avail.left()), avail.right() - self.width())
        y = min(max(global_pos.y(), avail.top()), avail.bottom() - self.height())
        return QPoint(x, y)

    def _refresh_list(self, text: str) -> None:
        self.list_widget.clear()
        needle = text.strip().lower()
        for label in self._classes:
            if needle and needle not in label.name.lower():
                continue
            item = QListWidgetItem()
            item.setData(ClassIdRole, label.id)
            self.list_widget.addItem(item)
            row = self._build_row(label)
            item.setSizeHint(row.sizeHint())
            self.list_widget.setItemWidget(item, row)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    @staticmethod
    def _build_row(label: ClassLabel) -> QWidget:
        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(f"background-color: {label.color}; border-radius: 6px; border: none;")

        name = QLabel(label.name)
        name.setStyleSheet(f"border: none; color: {TEXT_PRIMARY};")

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(6, 4, 6, 4)
        row_layout.setSpacing(8)
        row_layout.addWidget(swatch)
        row_layout.addWidget(name, stretch=1)
        return row

    def _accept_item(self, item: QListWidgetItem) -> None:
        class_id = item.data(ClassIdRole)
        if class_id:
            self.class_picked.emit(class_id)
        self.close()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.search_edit and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Down, Qt.Key_Up):
                count = self.list_widget.count()
                if count:
                    row = (self.list_widget.currentRow() + (1 if key == Qt.Key_Down else -1)) % count
                    self.list_widget.setCurrentRow(row)
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                item = self.list_widget.currentItem()
                if item is not None:
                    self._accept_item(item)
                return True
            if key == Qt.Key_Escape:
                self.close()
                return True
        return super().eventFilter(obj, event)
