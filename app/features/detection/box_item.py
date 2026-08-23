"""An interactive, resizable, movable bounding box on the detection canvas."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSceneContextMenuEvent,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QMenu,
    QStyleOptionGraphicsItem,
)

HANDLE_SIZE = 9.0
MIN_SIZE = 6.0

# Handle key -> which rect edges it drags. Order matches paint's handle list.
_HANDLES = ["tl", "t", "tr", "l", "r", "bl", "b", "br"]

_CURSOR_FOR_HANDLE = {
    "tl": Qt.SizeFDiagCursor,
    "br": Qt.SizeFDiagCursor,
    "tr": Qt.SizeBDiagCursor,
    "bl": Qt.SizeBDiagCursor,
    "t": Qt.SizeVerCursor,
    "b": Qt.SizeVerCursor,
    "l": Qt.SizeHorCursor,
    "r": Qt.SizeHorCursor,
}


class BoxItem(QGraphicsRectItem):
    """A bounding box. `rect()` is always in scene coordinates (pos stays at
    the origin), which keeps move/resize math simple and uniform."""

    def __init__(
        self,
        rect: QRectF,
        class_id: str,
        color: str,
        label: str,
        on_change: Callable[["BoxItem"], None],
        bounds: QRectF,
        on_edit_requested: Callable[["BoxItem"], None],
        on_delete_requested: Callable[["BoxItem"], None],
    ) -> None:
        super().__init__(rect)
        self.annotation_id: str | None = None
        self.class_id = class_id
        self.label = label
        self._color = QColor(color)
        self._on_change = on_change
        self._bounds = bounds
        self._on_edit_requested = on_edit_requested
        self._on_delete_requested = on_delete_requested
        self._mode: str | None = None  # "move" | a handle key | None
        self._press_rect = QRectF()
        self._press_pos = None

        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def set_label(self, label: str) -> None:
        self.label = label
        self.update()

    # -- painting -----------------------------------------------------

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self.rect().adjusted(-HANDLE_SIZE, -HANDLE_SIZE - 18, HANDLE_SIZE, HANDLE_SIZE)

    def shape(self) -> QPainterPath:  # noqa: N802
        # Include the (larger) handle hit-areas so corner/edge grabs near the
        # border are reliably detected, not just clicks inside the plain rect.
        path = QPainterPath()
        path.addRect(self.rect())
        if self.isSelected():
            for handle_rect in self._handle_rects().values():
                path.addRect(handle_rect.adjusted(-3, -3, 3, 3))
        return path

    def _display_scale(self) -> float:
        """The view's current zoom factor, floored at 1.0 — used to shrink
        (never grow) the label pill and selection handles so they stay a
        constant, reasonable size on screen when zoomed in a lot, instead of
        growing along with the image. Floored rather than fully compensated
        so zoomed-out behavior (already fine) is untouched, and boundingRect's
        fixed padding stays a safe upper bound at every zoom level."""
        scene = self.scene()
        views = scene.views() if scene is not None else []
        scale = views[0].transform().m11() if views else 1.0
        return max(scale, 1.0)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:  # noqa: N802
        rect = self.rect()
        selected = self.isSelected()
        s = self._display_scale()

        pen = QPen(self._color)
        pen.setWidth(3 if selected else 2)
        pen.setCosmetic(True)  # constant width in screen pixels, regardless of zoom
        painter.setPen(pen)
        fill = QColor(self._color)
        fill.setAlpha(50 if selected else 30)
        painter.setBrush(QBrush(fill))
        painter.drawRect(rect)

        font = QFont(painter.font())
        font.setBold(True)
        font.setPointSizeF(max(0.5, 9.0 / s))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        label_h = 18.0 / s
        label_w = metrics.horizontalAdvance(self.label) + 10.0 / s
        label_rect = QRectF(rect.left(), rect.top() - label_h, label_w, label_h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._color)
        painter.drawRect(label_rect)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(label_rect, Qt.AlignCenter, self.label)

        if selected:
            painter.setPen(Qt.NoPen)
            painter.setBrush(self._color)
            for handle_rect in self._handle_rects(HANDLE_SIZE / s).values():
                painter.drawRect(handle_rect)

    def _handle_rects(self, size: float = HANDLE_SIZE) -> dict[str, QRectF]:
        r = self.rect()
        h = size
        points = {
            "tl": r.topLeft(),
            "t": QRectF(r.left(), r.top(), r.width(), 0).center(),
            "tr": r.topRight(),
            "l": QRectF(r.left(), r.top(), 0, r.height()).center(),
            "r": QRectF(r.right(), r.top(), 0, r.height()).center(),
            "bl": r.bottomLeft(),
            "b": QRectF(r.left(), r.bottom(), r.width(), 0).center(),
            "br": r.bottomRight(),
        }
        return {key: QRectF(pt.x() - h / 2, pt.y() - h / 2, h, h) for key, pt in points.items()}

    def _handle_at(self, pos) -> str | None:
        for key, handle_rect in self._handle_rects().items():
            if handle_rect.adjusted(-3, -3, 3, 3).contains(pos):
                return key
        return None

    # -- interaction -----------------------------------------------

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # noqa: N802
        handle = self._handle_at(event.pos()) if self.isSelected() else None
        if handle:
            self.setCursor(QCursor(_CURSOR_FOR_HANDLE[handle]))
        elif self.rect().contains(event.pos()):
            self.setCursor(QCursor(Qt.SizeAllCursor))
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        if self.scene() is not None:
            for other in self.scene().selectedItems():
                if other is not self:
                    other.setSelected(False)
        self.setSelected(True)

        handle = self._handle_at(event.pos())
        self._mode = handle if handle else "move"
        self._press_rect = QRectF(self.rect())
        self._press_pos = event.pos()
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._mode is None or self._press_pos is None:
            return
        delta = event.pos() - self._press_pos
        if self._mode == "move":
            new_rect = self._press_rect.translated(delta)
            new_rect = self._clamp_translate(new_rect)
        else:
            new_rect = self._resize(self._mode, self._press_rect, delta)
        self.prepareGeometryChange()
        self.setRect(new_rect)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._mode is not None:
            self._mode = None
            self._on_change(self)
        event.accept()

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent) -> None:  # noqa: N802
        if self.scene() is not None:
            for other in self.scene().selectedItems():
                if other is not self:
                    other.setSelected(False)
        self.setSelected(True)

        menu = QMenu()
        edit_action = menu.addAction("Edit Class…")
        delete_action = menu.addAction("Delete Box")
        chosen = menu.exec(event.screenPos())
        if chosen == edit_action:
            self._on_edit_requested(self)
        elif chosen == delete_action:
            self._on_delete_requested(self)
        event.accept()

    def _clamp_translate(self, rect: QRectF) -> QRectF:
        dx = dy = 0.0
        if rect.left() < self._bounds.left():
            dx = self._bounds.left() - rect.left()
        elif rect.right() > self._bounds.right():
            dx = self._bounds.right() - rect.right()
        if rect.top() < self._bounds.top():
            dy = self._bounds.top() - rect.top()
        elif rect.bottom() > self._bounds.bottom():
            dy = self._bounds.bottom() - rect.bottom()
        return rect.translated(dx, dy)

    def _resize(self, handle: str, start: QRectF, delta) -> QRectF:
        r = QRectF(start)
        if "l" in handle:
            r.setLeft(min(start.right() - MIN_SIZE, max(self._bounds.left(), start.left() + delta.x())))
        if "r" in handle:
            r.setRight(max(start.left() + MIN_SIZE, min(self._bounds.right(), start.right() + delta.x())))
        if "t" in handle:
            r.setTop(min(start.bottom() - MIN_SIZE, max(self._bounds.top(), start.top() + delta.y())))
        if "b" in handle:
            r.setBottom(max(start.top() + MIN_SIZE, min(self._bounds.bottom(), start.bottom() + delta.y())))
        return r.normalized()
