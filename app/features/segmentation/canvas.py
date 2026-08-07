"""Interactive polygon annotation canvas.

Click on empty image area to start a polygon, click again to add each
vertex, then either double-click, press Enter, or click back near the
first point to close it. Closing a shape opens a search-as-you-type class
picker anchored on it — there's no "active class" step first, since a
single image can hold objects of many different classes at once and a
global active class can't represent that. Existing polygons can be
selected, dragged as a whole, have vertices dragged individually, or a
vertex removed with a double-click; right-click one to edit its class or
delete it. Delete removes the selected polygon. Escape cancels an
in-progress polygon (or a class picker that's still open). Objects nest
and overlap in real images, so a plain click that lands on an existing
shape selects/drags it instead of starting a new one — hold Shift on that
first click to force a new shape there anyway (drawing one inside/over
another). Every committed change is persisted immediately.

While drawing, each placed point is numbered and the first point turns
green (with a live status message) once you're close enough to click it
and close the shape — the two cues a first-time segmentation user needs
most: "where am I in the sequence" and "how do I finish".
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsSimpleTextItem

from app.features.project.manager import ProjectController
from app.features.segmentation.polygon_item import MIN_VERTICES, PolygonItem
from app.shared.labeling.class_picker import ClassPickerPopup
from app.shared.labeling.zoomable_view import ZoomableGraphicsView

# Bright green: visible while drawing a polygon over dark image regions,
# where the previous near-black pen effectively disappeared.
_DRAW_PEN_COLOR = "#4ade80"
_CLOSE_SNAP_PIXELS = 10
_VERTEX_MARKER_RADIUS = 3.5
_CLOSE_HIGHLIGHT_COLOR = "#10b981"  # matches the app's general "complete" green


class SegmentationCanvas(ZoomableGraphicsView):
    status_changed = Signal(str)  # live "what's happening right now" hint

    def __init__(self, controller: ProjectController) -> None:
        super().__init__()
        self._controller = controller
        self._image_id: str | None = None
        self._polygons: dict[str, PolygonItem] = {}

        self._drawing_points: list[QPointF] = []
        self._preview_item: QGraphicsPolygonItem | None = None
        self._preview_markers: list[QGraphicsEllipseItem] = []
        self._preview_labels: list[QGraphicsSimpleTextItem] = []
        self._rubber_line_end: QPointF | None = None
        self._pending_picker: ClassPickerPopup | None = None

        self.setMouseTracking(True)
        # Always ready to draw — there's no "active class" gate anymore.
        self.viewport().setCursor(Qt.CrossCursor)

    def refresh_status(self) -> None:
        if self.is_drawing:
            return  # mid-draw status is driven by _rebuild_preview instead
        self.status_changed.emit("Click on the image to start outlining an object.")

    @property
    def is_drawing(self) -> bool:
        return bool(self._drawing_points)

    def load_image(self, image_id: str | None) -> None:
        self._cancel_drawing()
        self._image_id = image_id
        self._polygons = {}
        project = self._controller.project
        entry = project.find_image(image_id) if project and image_id else None
        pixmap = self._load_pixmap(entry.path if entry else None)
        if pixmap is None or entry is None:
            return
        width, height = pixmap.width(), pixmap.height()
        for annotation in entry.annotations:
            if len(annotation.points) < MIN_VERTICES:
                continue
            points = [QPointF(x * width, y * height) for x, y in annotation.points]
            self._add_polygon_item(annotation.id, annotation.class_id, points)

    def refresh_class_styles(self) -> None:
        project = self._controller.project
        if project is None:
            return
        for item in self._polygons.values():
            label = project.find_class(item.class_id)
            if label is not None:
                item.set_color(label.color)
                item.set_label(label.name)

    def select_annotation(self, annotation_id: str) -> None:
        item = self._polygons.get(annotation_id)
        if item is None:
            return
        self._scene.clearSelection()
        item.setSelected(True)
        self.centerOn(item)

    def update_item_class(self, annotation_id: str, class_id: str) -> None:
        item = self._polygons.get(annotation_id)
        project = self._controller.project
        if item is None or project is None:
            return
        label = project.find_class(class_id)
        item.class_id = class_id
        item.set_color(label.color if label else "#888888")
        item.set_label(label.name if label else "?")

    def remove_item(self, annotation_id: str) -> None:
        item = self._polygons.pop(annotation_id, None)
        if item is not None:
            self._scene.removeItem(item)

    def _add_polygon_item(self, annotation_id: str, class_id: str, points: list[QPointF]) -> PolygonItem:
        project = self._controller.project
        label = project.find_class(class_id) if project else None
        color = label.color if label else "#888888"
        name = label.name if label else "?"
        bounds = self._pixmap_item.boundingRect() if self._pixmap_item else QRectF()
        item = PolygonItem(
            points, class_id, color, name, self._on_polygon_changed, bounds, self._on_polygon_edit_requested, self._delete_item
        )
        item.annotation_id = annotation_id
        self._scene.addItem(item)
        self._polygons[annotation_id] = item
        return item

    def _on_polygon_changed(self, item: PolygonItem) -> None:
        if not self._image_id or not item.annotation_id:
            return
        points = self._polygon_to_points(item.polygon())
        self._controller.update_annotation(self._image_id, item.annotation_id, points=points)

    def _polygon_to_points(self, polygon: QPolygonF) -> list[tuple[float, float]]:
        if self._pixmap_item is None:
            return []
        size = self._pixmap_item.pixmap().size()
        width, height = max(1, size.width()), max(1, size.height())
        return [(p.x() / width, p.y() / height) for p in polygon]

    # -- drawing new polygons -----------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._pixmap_item is not None:
            item = self.itemAt(event.pos())
            clicked_empty = item is None or item is self._pixmap_item
            scene_pos = self.mapToScene(event.pos())
            inside_image = self._pixmap_item.boundingRect().contains(scene_pos)

            if self.is_drawing and inside_image:
                self._add_draw_point(scene_pos, event.pos())
                event.accept()
                return
            # Objects nest and overlap in real images, so a click landing on
            # an existing shape can't always mean "start a new one" — that
            # would make it impossible to select/move/drag-a-vertex-of the
            # shape underneath. Shift is the explicit "yes, draw here
            # anyway" override, so a shape can still be drawn inside/over
            # another one.
            force_new = bool(event.modifiers() & Qt.ShiftModifier)
            if (clicked_empty or force_new) and inside_image:
                self._drawing_points = [scene_pos]
                self._rebuild_preview()
                event.accept()
                return
        if event.button() == Qt.RightButton and self.is_drawing:
            self._finish_polygon()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.is_drawing:
            self._rubber_line_end = self.mapToScene(event.pos())
            self._rebuild_preview()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.is_drawing:
            self._finish_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _is_near_first_point(self, scene_pos: QPointF, viewport_pos) -> bool:
        """Whether scene_pos/viewport_pos is close enough to the first
        drawn point to count as "click here to close the shape"."""
        if len(self._drawing_points) < MIN_VERTICES:
            return False
        first_viewport = self.mapFromScene(self._drawing_points[0])
        return (viewport_pos - first_viewport).manhattanLength() <= _CLOSE_SNAP_PIXELS * 2

    def _add_draw_point(self, scene_pos: QPointF, viewport_pos) -> None:
        if self._is_near_first_point(scene_pos, viewport_pos):
            self._finish_polygon()
            return
        self._drawing_points.append(scene_pos)
        self._rebuild_preview()

    def _rebuild_preview(self) -> None:
        self._clear_preview_items()
        if not self._drawing_points:
            return
        points = list(self._drawing_points)
        if self._rubber_line_end is not None:
            points = points + [self._rubber_line_end]
        pen = QPen(QColor(_DRAW_PEN_COLOR))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(2)
        self._preview_item = self._scene.addPolygon(QPolygonF(points), pen)

        near_close = False
        if self._rubber_line_end is not None:
            viewport_pos = self.mapFromScene(self._rubber_line_end)
            near_close = self._is_near_first_point(self._rubber_line_end, viewport_pos)

        for index, point in enumerate(self._drawing_points):
            highlight = index == 0 and near_close
            radius = _VERTEX_MARKER_RADIUS * (1.7 if highlight else 1.0)
            outline = QColor(_CLOSE_HIGHLIGHT_COLOR) if highlight else QColor(_DRAW_PEN_COLOR)
            fill = QColor(_CLOSE_HIGHLIGHT_COLOR) if highlight else QColor("#ffffff")
            marker_pen = QPen(outline)
            marker_pen.setWidth(2 if highlight else 1)
            marker = self._scene.addEllipse(
                point.x() - radius, point.y() - radius, radius * 2, radius * 2, marker_pen, fill
            )
            marker.setZValue(2)
            self._preview_markers.append(marker)

            label = self._scene.addSimpleText(str(index + 1))
            font = label.font()
            font.setPointSize(7)
            font.setBold(True)
            label.setFont(font)
            label.setBrush(QColor(_DRAW_PEN_COLOR))
            label.setZValue(2)
            label_rect = label.boundingRect()
            label.setPos(point.x() - label_rect.width() / 2, point.y() - radius - label_rect.height() - 3)
            self._preview_labels.append(label)

        if near_close:
            self.status_changed.emit("Click here to close the shape (or press Enter / double-click).")
        else:
            count = len(self._drawing_points)
            self.status_changed.emit(
                f"Point {count} placed — keep clicking around the object, "
                "then click back on point 1 (or press Enter) to close it."
            )

    def _clear_preview_items(self) -> None:
        if self._preview_item is not None:
            self._scene.removeItem(self._preview_item)
            self._preview_item = None
        for marker in self._preview_markers:
            self._scene.removeItem(marker)
        self._preview_markers = []
        for label in self._preview_labels:
            self._scene.removeItem(label)
        self._preview_labels = []

    def _finish_polygon(self) -> None:
        if len(self._drawing_points) < MIN_VERTICES:
            self.status_changed.emit(f"Need at least {MIN_VERTICES} points before you can close a shape — keep clicking.")
            return
        if not self._image_id:
            self._cancel_drawing()
            return

        points = self._drawing_points
        self._drawing_points = []
        self._rubber_line_end = None
        self._clear_preview_items()

        # Keep a static (non-rubber-banded) outline of the closed shape on
        # screen — not yet a real annotation — while the class picker is
        # open, so it doesn't visually vanish mid-pick.
        pen = QPen(QColor(_DRAW_PEN_COLOR))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(2)
        self._preview_item = self._scene.addPolygon(QPolygonF(points), pen)

        self.status_changed.emit("Pick a class for this shape…")
        self._prompt_class_for_new_polygon(points)

    def _prompt_class_for_new_polygon(self, points: list[QPointF]) -> None:
        project = self._controller.project
        if project is None or not project.classes or self._image_id is None:
            self._cancel_drawing()
            return
        anchor_scene = QPolygonF(points).boundingRect().topLeft()
        anchor = self.mapToGlobal(self.mapFromScene(anchor_scene))
        popup = ClassPickerPopup(project.classes, self)
        self._pending_picker = popup
        popup.class_picked.connect(lambda class_id, pts=points: self._commit_new_polygon(pts, class_id))
        popup.destroyed.connect(self._on_picker_closed)
        popup.popup_at(anchor)

    def _on_picker_closed(self) -> None:
        # Catches every dismissal path (Escape, click-outside, a class
        # actually picked) — if the pending outline is still sitting there
        # un-persisted at this point, the popup closed without a pick, so
        # discard it rather than leaving an unclassed shape behind.
        self._pending_picker = None
        if self._preview_item is not None:
            self._clear_preview_items()
        self.refresh_status()

    def _commit_new_polygon(self, points: list[QPointF], class_id: str) -> None:
        self._clear_preview_items()
        if not self._image_id:
            return
        annotation = self._controller.add_annotation(self._image_id, class_id, self._polygon_to_points(QPolygonF(points)))
        item = self._add_polygon_item(annotation.id, class_id, points)
        self._scene.clearSelection()
        item.setSelected(True)
        self.refresh_status()

    def _cancel_drawing(self) -> None:
        if self._pending_picker is not None:
            self._pending_picker.close()
            return
        had_progress = self.is_drawing
        self._drawing_points = []
        self._rubber_line_end = None
        self._clear_preview_items()
        if had_progress:
            self.refresh_status()

    # -- editing/deleting existing polygons -----------------------------------

    def _on_polygon_edit_requested(self, item: PolygonItem) -> None:
        project = self._controller.project
        if project is None or not project.classes:
            return
        anchor_scene = item.polygon().boundingRect().topLeft()
        anchor = self.mapToGlobal(self.mapFromScene(anchor_scene))
        popup = ClassPickerPopup(project.classes, self)
        annotation_id = item.annotation_id
        popup.class_picked.connect(lambda class_id, aid=annotation_id: self._reclassify_polygon(aid, class_id))
        popup.popup_at(anchor)

    def _reclassify_polygon(self, annotation_id: str | None, class_id: str) -> None:
        if not annotation_id or not self._image_id:
            return
        self._controller.update_annotation(self._image_id, annotation_id, class_id=class_id)
        self.update_item_class(annotation_id, class_id)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape and self.is_drawing:
            self._cancel_drawing()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.is_drawing:
            self._finish_polygon()
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_selected()
            return
        super().keyPressEvent(event)

    def _delete_selected(self) -> None:
        selected = [item for item in self._scene.selectedItems() if isinstance(item, PolygonItem)]
        for item in selected:
            self._delete_item(item)

    def _delete_item(self, item: PolygonItem) -> None:
        annotation_id = item.annotation_id
        self._scene.removeItem(item)
        if annotation_id:
            self._polygons.pop(annotation_id, None)
            self._controller.remove_annotation(self._image_id, annotation_id)
