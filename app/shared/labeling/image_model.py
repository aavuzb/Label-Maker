"""Qt model exposing the open project's images to list/grid views."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtCore import QAbstractListModel

from app.features.project.manager import ProjectController
from app.shared.thumbnails import ThumbnailCache

ImageIdRole = Qt.UserRole + 1
ClassIdRole = Qt.UserRole + 2
ClassColorRole = Qt.UserRole + 3
StateRole = Qt.UserRole + 4
PathRole = Qt.UserRole + 5
AnnotationCountRole = Qt.UserRole + 6
AnnotationClassIdsRole = Qt.UserRole + 7
ClassNameRole = Qt.UserRole + 8

THUMBNAIL_SIZE = 112
MIN_THUMBNAIL_SIZE = 64
MAX_THUMBNAIL_SIZE = 256


class ImageListModel(QAbstractListModel):
    def __init__(self, controller: ProjectController) -> None:
        super().__init__()
        self._controller = controller
        self._thumbnails = ThumbnailCache()
        self._thumbnail_size = THUMBNAIL_SIZE

        controller.project_opened.connect(self._reset)
        controller.project_closed.connect(self._reset)
        controller.images_added.connect(self._on_images_added)
        controller.images_removed.connect(self._reset)
        controller.image_updated.connect(self._on_image_updated)
        controller.classes_changed.connect(self._on_classes_changed)

    def set_thumbnail_size(self, size: int) -> None:
        """Changes the resolution thumbnails are decoded/rendered at (grid zoom)."""
        if size == self._thumbnail_size:
            return
        self._thumbnail_size = size
        if self.rowCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, 0), [Qt.DecorationRole])

    def _reset(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def _on_images_added(self, added: list) -> None:
        project = self._controller.project
        if project is None:
            return
        start = len(project.images) - len(added)
        self.beginInsertRows(QModelIndex(), start, len(project.images) - 1)
        self.endInsertRows()

    def _on_image_updated(self, image_id: str) -> None:
        row = self._row_for_id(image_id)
        if row is None:
            return
        index = self.index(row, 0)
        self.dataChanged.emit(index, index)

    def _on_classes_changed(self) -> None:
        if self.rowCount() == 0:
            return
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, 0))

    def _row_for_id(self, image_id: str) -> int | None:
        project = self._controller.project
        if project is None:
            return None
        for row, entry in enumerate(project.images):
            if entry.id == image_id:
                return row
        return None

    # -- QAbstractListModel ------------------------------------------------

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        project = self._controller.project
        return len(project.images) if project else 0

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        project = self._controller.project
        if project is None or not index.isValid():
            return None
        entry = project.images[index.row()]

        if role == Qt.DisplayRole:
            return Path(entry.path).name
        if role == Qt.ToolTipRole:
            return entry.path
        if role == Qt.DecorationRole:
            return self._thumbnails.get(entry.path, self._thumbnail_size)
        if role == ImageIdRole:
            return entry.id
        if role == PathRole:
            return entry.path
        if role == ClassIdRole:
            return entry.class_id
        if role == ClassColorRole:
            if entry.class_id is not None:
                label = project.find_class(entry.class_id)
                return label.color if label else None
            # Detection/segmentation: one image can hold several objects of
            # several different classes, so picking any one class's color
            # to border the whole thumbnail with — even when every object
            # on it currently happens to share a class — reads as "this
            # image IS that class", which isn't true and stops being true
            # the moment a second, differently-classed object is added.
            # None here always falls through to the delegate's single
            # generic "annotated" color instead.
            return None
        if role == ClassNameRole:
            # Whole-image classification only: detection/segmentation
            # images can carry several annotations at once, so a single
            # "this image is class X" caption doesn't apply to them.
            if entry.class_id is not None:
                label = project.find_class(entry.class_id)
                return label.name if label else None
            return None
        if role == StateRole:
            return entry.state
        if role == AnnotationCountRole:
            return len(entry.annotations)
        if role == AnnotationClassIdsRole:
            return [a.class_id for a in entry.annotations]
        return None


class ImageFilterProxyModel(QSortFilterProxyModel):
    FILTER_ALL = "all"
    FILTER_UNLABELED = "unlabeled"
    FILTER_CLASS = "class"

    def __init__(self) -> None:
        super().__init__()
        self._mode = self.FILTER_ALL
        self._class_id: str | None = None

    def set_filter(self, mode: str, class_id: str | None = None) -> None:
        self._mode = mode
        self._class_id = class_id
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        if self._mode == self.FILTER_ALL:
            return True
        index = self.sourceModel().index(source_row, 0, source_parent)
        if self._mode == self.FILTER_UNLABELED:
            return index.data(StateRole) == "unlabeled"
        if self._mode == self.FILTER_CLASS:
            if index.data(ClassIdRole) == self._class_id:
                return True
            return self._class_id in (index.data(AnnotationClassIdsRole) or [])
        return True
