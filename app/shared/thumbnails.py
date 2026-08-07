"""Lightweight, size-capped thumbnail cache backed by QImageReader."""

from __future__ import annotations

from collections import OrderedDict

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImageReader, QPixmap

_MAX_CACHE_ENTRIES = 4000


class ThumbnailCache:
    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[str, int], QPixmap] = OrderedDict()

    def get(self, path: str, size: int) -> QPixmap:
        key = (path, size)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        pixmap = self._load(path, size)
        self._cache[key] = pixmap
        if len(self._cache) > _MAX_CACHE_ENTRIES:
            self._cache.popitem(last=False)
        return pixmap

    def invalidate(self, path: str) -> None:
        for key in [k for k in self._cache if k[0] == path]:
            del self._cache[key]

    @staticmethod
    def _load(path: str, size: int) -> QPixmap:
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        original = reader.size()
        if original.isValid() and not original.isEmpty():
            scaled = original.scaled(QSize(size, size), Qt.KeepAspectRatio)
            reader.setScaledSize(scaled)
        image = reader.read()
        if image.isNull():
            pixmap = QPixmap(size, size)
            pixmap.fill()
            return pixmap
        return QPixmap.fromImage(image)
