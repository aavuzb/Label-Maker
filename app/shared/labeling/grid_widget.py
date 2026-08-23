"""Thumbnail grid supporting ctrl/shift/drag multi-selection."""

from __future__ import annotations

from PySide6.QtCore import QItemSelection, QModelIndex, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QAbstractItemView, QListView, QMenu

from app.shared.labeling.delegate import ThumbnailDelegate
from app.shared.labeling.image_model import THUMBNAIL_SIZE, ImageIdRole


class ImageGridWidget(QListView):
    assign_requested = Signal()
    remove_requested = Signal()
    edit_requested = Signal(QPoint)  # global position to anchor the class picker popup
    zoom_requested = Signal(int)  # +1 to zoom in, -1 to zoom out (Ctrl+wheel)
    selection_changed = Signal(list)  # list[str] image ids
    current_changed = Signal(str)  # image id (or "" when cleared)

    def __init__(self, show_edit_action: bool = False) -> None:
        super().__init__()
        self._show_edit_action = show_edit_action
        self.setViewMode(QListView.IconMode)
        self.setMovement(QListView.Static)
        self.setResizeMode(QListView.Adjust)
        self.setUniformItemSizes(True)
        self.setSpacing(4)
        self.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setItemDelegate(ThumbnailDelegate(self))
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        tooltip = "Click to select · Ctrl/Shift-click for multiple · Delete or right-click to remove from project · Ctrl+Scroll to zoom"
        if show_edit_action:
            tooltip = "Click to select · Ctrl/Shift-click for multiple · Right-click to edit class or remove · Ctrl+Scroll to zoom"
        self.setToolTip(tooltip)

        QShortcut(QKeySequence.SelectAll, self, activated=self.selectAll)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            self.zoom_requested.emit(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def _show_context_menu(self, pos) -> None:
        count = len(self.selected_image_ids())
        if count == 0:
            return
        menu = QMenu(self)
        edit_action = None
        if self._show_edit_action:
            edit_label = "Edit Class…" if count == 1 else f"Edit Class ({count})…"
            edit_action = menu.addAction(edit_label)
        remove_label = "Remove Image" if count == 1 else f"Remove {count} Images"
        remove_action = menu.addAction(f"{remove_label} from Project…")
        global_pos = self.viewport().mapToGlobal(pos)
        chosen = menu.exec(global_pos)
        if edit_action is not None and chosen == edit_action:
            self.edit_requested.emit(global_pos)
        elif chosen == remove_action:
            self.remove_requested.emit()

    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection) -> None:  # noqa: N802
        super().selectionChanged(selected, deselected)
        self.selection_changed.emit(self.selected_image_ids())

    def currentChanged(self, current: QModelIndex, previous: QModelIndex) -> None:  # noqa: N802
        super().currentChanged(current, previous)
        self.current_changed.emit(current.data(ImageIdRole) or "" if current.isValid() else "")

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.assign_requested.emit()
            return
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.selected_image_ids():
                self.remove_requested.emit()
            return
        if key == Qt.Key_D:
            self._step_current(1)
            return
        if key == Qt.Key_A and not (event.modifiers() & Qt.ControlModifier):
            self._step_current(-1)
            return
        super().keyPressEvent(event)

    def _step_current(self, delta: int) -> None:
        model = self.model()
        if model is None or model.rowCount() == 0:
            return
        current = self.currentIndex()
        row = current.row() + delta if current.isValid() else 0
        row = max(0, min(model.rowCount() - 1, row))
        new_index = model.index(row, 0)
        self.setCurrentIndex(new_index)
        self.selectionModel().select(
            new_index,
            self.selectionModel().SelectionFlag.ClearAndSelect,
        )
        self.scrollTo(new_index)

    def selected_image_ids(self) -> list[str]:
        return [index.data(ImageIdRole) for index in self.selectionModel().selectedIndexes()]

    def current_image_id(self) -> str | None:
        index = self.currentIndex()
        return index.data(ImageIdRole) if index.isValid() else None

    def neighbor_image_id(self, delta: int) -> str | None:
        """The image id ± delta rows away from the current row, in the same
        (filtered) display order _step_current walks — or None past either
        end."""
        model = self.model()
        current = self.currentIndex()
        if model is None or not current.isValid():
            return None
        row = current.row() + delta
        if 0 <= row < model.rowCount():
            return model.index(row, 0).data(ImageIdRole)
        return None
