"""Base QGraphicsView: loads an image, supports fit-to-screen and zoom.

Shared by the plain preview (classification) and the interactive
annotation canvases (detection/segmentation), which layer drawing and
editing interactions on top of this.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

ZOOM_FACTOR = 1.25
MIN_SCALE = 0.05
MAX_SCALE = 12.0


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._fit_mode = True
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.darkGray)

    @property
    def has_image(self) -> bool:
        return self._pixmap_item is not None

    @property
    def image_size(self):
        return self._pixmap_item.pixmap().size() if self._pixmap_item else None

    def _load_pixmap(self, path: str | None) -> QPixmap | None:
        """Clears the scene and loads `path` as the background image. Subclasses
        call this, then re-populate any annotation items on top of it."""
        self._scene.clear()
        self._pixmap_item = None
        if not path:
            return None
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return None
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(-1)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        if self._fit_mode:
            self.fit_to_screen()
        return pixmap

    def fit_to_screen(self) -> None:
        if self._pixmap_item is None:
            return
        self._fit_mode = True
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def zoom_in(self) -> None:
        self._zoom(ZOOM_FACTOR)

    def zoom_out(self) -> None:
        self._zoom(1 / ZOOM_FACTOR)

    def _zoom(self, factor: float) -> None:
        if self._pixmap_item is None:
            return
        target_scale = self.transform().m11() * factor
        if target_scale < MIN_SCALE or target_scale > MAX_SCALE:
            return
        self._fit_mode = False
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit_mode and self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._pixmap_item is None:
            return
        self._zoom(ZOOM_FACTOR if event.angleDelta().y() > 0 else 1 / ZOOM_FACTOR)
