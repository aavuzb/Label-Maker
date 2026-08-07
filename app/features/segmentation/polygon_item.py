"""An interactive polygon (vertex-editable, movable) on the segmentation canvas."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsSceneContextMenuEvent,
    QGraphicsSceneMouseEvent,
    QMenu,
    QStyleOptionGraphicsItem,
)

VERTEX_RADIUS = 5.0
MIN_VERTICES = 3


class PolygonItem(QGraphicsPolygonItem):
    """`polygon()` points are always in scene coordinates (pos stays at origin)."""

    def __init__(
        self,
        points: list[QPointF],
        class_id: str,
        color: str,
        label: str,
        on_change: Callable[["PolygonItem"], None],
        bounds: QRectF,
        on_edit_requested: Callable[["PolygonItem"], None],
        on_delete_requested: Callable[["PolygonItem"], None],
    ) -> None:
        super().__init__(QPolygonF(points))
        self.annotation_id: str | None = None
        self.class_id = class_id
        self.label = label
        self._color = QColor(color)
        self._on_change = on_change
        self._bounds = bounds
        self._on_edit_requested = on_edit_requested
        self._on_delete_requested = on_delete_requested
        self._drag_vertex: int | None = None
        self._mode: str | None = None  # "vertex" | "move" | None
        self._press_pos: QPointF | None = None
        self._press_polygon: QPolygonF | None = None

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
        return self.polygon().boundingRect().adjusted(-VERTEX_RADIUS * 2, -VERTEX_RADIUS * 2 - 18, VERTEX_RADIUS * 2, VERTEX_RADIUS * 2)

    def shape(self) -> QPainterPath:  # noqa: N802
        path = QPainterPath()
        path.addPolygon(self.polygon())
        path.closeSubpath()
        if self.isSelected():
            for point in self.polygon():
                path.addEllipse(point, VERTEX_RADIUS + 3, VERTEX_RADIUS + 3)
        return path

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:  # noqa: N802
        polygon = self.polygon()
        selected = self.isSelected()

        pen = QPen(self._color)
        pen.setWidth(3 if selected else 2)
        painter.setPen(pen)
        fill = QColor(self._color)
        fill.setAlpha(60 if selected else 35)
        painter.setBrush(QBrush(fill))
        painter.drawPolygon(polygon)

        if not polygon.isEmpty():
            anchor = polygon.boundingRect().topLeft()
            font = QFont(painter.font())
            font.setBold(True)
            font.setPointSize(9)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            label_w = metrics.horizontalAdvance(self.label) + 10
            label_rect = QRectF(anchor.x(), anchor.y() - 18, label_w, 18)
            painter.setPen(Qt.NoPen)
            painter.setBrush(self._color)
            painter.drawRect(label_rect)
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(label_rect, Qt.AlignCenter, self.label)

        if selected:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#ffffff"))
            for point in polygon:
                painter.drawEllipse(point, VERTEX_RADIUS, VERTEX_RADIUS)
            painter.setPen(QPen(self._color, 2))
            painter.setBrush(Qt.NoBrush)
            for point in polygon:
                painter.drawEllipse(point, VERTEX_RADIUS, VERTEX_RADIUS)

    # -- interaction -----------------------------------------------

    def _vertex_at(self, pos: QPointF) -> int | None:
        for index, point in enumerate(self.polygon()):
            if (point - pos).manhattanLength() <= VERTEX_RADIUS * 2.5:
                return index
        return None

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        if self.scene() is not None:
            for other in self.scene().selectedItems():
                if other is not self:
                    other.setSelected(False)
        was_selected = self.isSelected()
        self.setSelected(True)

        vertex = self._vertex_at(event.pos()) if was_selected else None
        self._mode = "vertex" if vertex is not None else "move"
        self._drag_vertex = vertex
        self._press_pos = event.pos()
        self._press_polygon = QPolygonF(self.polygon())
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._mode is None or self._press_pos is None or self._press_polygon is None:
            return
        delta = event.pos() - self._press_pos
        polygon = QPolygonF(self._press_polygon)
        if self._mode == "vertex" and self._drag_vertex is not None:
            point = self._press_polygon[self._drag_vertex] + delta
            point = self._clamp_point(point)
            polygon[self._drag_vertex] = point
        else:
            polygon = QPolygonF([self._clamp_point(p + delta) for p in self._press_polygon])
        self.prepareGeometryChange()
        self.setPolygon(polygon)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._mode is not None:
            self._mode = None
            self._drag_vertex = None
            self._on_change(self)
        event.accept()

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        vertex = self._vertex_at(event.pos())
        if vertex is not None and self.polygon().size() > MIN_VERTICES:
            polygon = QPolygonF(self.polygon())
            polygon.remove(vertex)
            self.prepareGeometryChange()
            self.setPolygon(polygon)
            self.update()
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
        delete_action = menu.addAction("Delete Shape")
        chosen = menu.exec(event.screenPos())
        if chosen == edit_action:
            self._on_edit_requested(self)
        elif chosen == delete_action:
            self._on_delete_requested(self)
        event.accept()

    def _clamp_point(self, point: QPointF) -> QPointF:
        x = min(max(point.x(), self._bounds.left()), self._bounds.right())
        y = min(max(point.y(), self._bounds.top()), self._bounds.bottom())
        return QPointF(x, y)
