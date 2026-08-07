"""Interactive bounding-box annotation canvas.

Click-drag on empty image area draws a new box; once you release the
mouse, a class picker (search-as-you-type, same widget the grid's "Edit
Class…" uses) appears anchored at the box's top-left corner so you choose
what it's tagging — there's no "active class" step first, since a single
image can hold objects of many different classes at once and a global
active class can't represent that. Click an existing box to select/move/
resize it; right-click it to edit its class or delete it. Objects nest and
overlap in real images, so a plain click-drag that lands on an existing
box moves/resizes it instead of drawing — hold Shift while dragging to
force a new box there anyway (drawing one inside/over another). Every
committed change (drawn, moved, resized, reclassified, deleted) is
persisted through the ProjectController immediately.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPen
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsView

from app.features.detection.box_item import BoxItem
from app.features.project.manager import ProjectController
from app.shared.labeling.class_picker import ClassPickerPopup
from app.shared.labeling.zoomable_view import ZoomableGraphicsView

MIN_DRAW_SIZE = 4.0
# Bright green: visible while dragging out a box over dark image regions,
# where the previous near-black pen effectively disappeared.
_DRAW_PEN_COLOR = "#4ade80"


class DetectionCanvas(ZoomableGraphicsView):
    status_changed = Signal(str)  # live "what's happening right now" hint

    def __init__(self, controller: ProjectController) -> None:
        super().__init__()
        self._controller = controller
        self._image_id: str | None = None
        self._boxes: dict[str, BoxItem] = {}
        self._draw_item: QGraphicsRectItem | None = None
        self._draw_start = None
        self._pending_picker: ClassPickerPopup | None = None
        self.setDragMode(QGraphicsView.NoDrag)
        # Always ready to draw — there's no "active class" gate anymore.
        self.viewport().setCursor(Qt.CrossCursor)

    def refresh_status(self) -> None:
        self.status_changed.emit("Click and drag on the image to draw a box, then pick its class.")

    def load_image(self, image_id: str | None) -> None:
        self._cancel_pending_box()
        self._image_id = image_id
        self._boxes = {}
        project = self._controller.project
        entry = project.find_image(image_id) if project and image_id else None
        pixmap = self._load_pixmap(entry.path if entry else None)
        if pixmap is None or entry is None:
            return
        width, height = pixmap.width(), pixmap.height()
        for annotation in entry.annotations:
            if len(annotation.points) != 2:
                continue
            (x1, y1), (x2, y2) = annotation.points
            rect = QRectF(x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height).normalized()
            self._add_box_item(annotation.id, annotation.class_id, rect)

    def refresh_class_styles(self) -> None:
        """Re-reads class color/name (e.g. after a rename) onto existing boxes."""
        project = self._controller.project
        if project is None:
            return
        for item in self._boxes.values():
            label = project.find_class(item.class_id)
            if label is not None:
                item.set_color(label.color)
                item.set_label(label.name)

    def select_annotation(self, annotation_id: str) -> None:
        item = self._boxes.get(annotation_id)
        if item is None:
            return
        self._scene.clearSelection()
        item.setSelected(True)
        self.centerOn(item)

    def update_item_class(self, annotation_id: str, class_id: str) -> None:
        item = self._boxes.get(annotation_id)
        project = self._controller.project
        if item is None or project is None:
            return
        label = project.find_class(class_id)
        item.class_id = class_id
        item.set_color(label.color if label else "#888888")
        item.set_label(label.name if label else "?")

    def remove_item(self, annotation_id: str) -> None:
        item = self._boxes.pop(annotation_id, None)
        if item is not None:
            self._scene.removeItem(item)

    def _add_box_item(self, annotation_id: str, class_id: str, rect: QRectF) -> BoxItem:
        project = self._controller.project
        label = project.find_class(class_id) if project else None
        color = label.color if label else "#888888"
        name = label.name if label else "?"
        bounds = self._pixmap_item.boundingRect() if self._pixmap_item else QRectF(rect)
        item = BoxItem(rect, class_id, color, name, self._on_box_changed, bounds, self._on_box_edit_requested, self._delete_item)
        item.annotation_id = annotation_id
        self._scene.addItem(item)
        self._boxes[annotation_id] = item
        return item

    def _on_box_changed(self, item: BoxItem) -> None:
        if not self._image_id or not item.annotation_id:
            return
        self._controller.update_annotation(self._image_id, item.annotation_id, points=self._rect_to_points(item.rect()))

    def _rect_to_points(self, rect: QRectF) -> list[tuple[float, float]]:
        if self._pixmap_item is None:
            return []
        size = self._pixmap_item.pixmap().size()
        width, height = max(1, size.width()), max(1, size.height())
        return [(rect.left() / width, rect.top() / height), (rect.right() / width, rect.bottom() / height)]

    # -- drawing new boxes -----------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._image_id and self._pixmap_item is not None:
            item = self.itemAt(event.pos())
            scene_pos = self.mapToScene(event.pos())
            clicked_empty = item is None or item is self._pixmap_item
            # Objects nest and overlap in real images, so a click landing on
            # an existing box can't always mean "start a new one" — that
            # would make it impossible to select/move/resize the box
            # underneath. Shift is the explicit "yes, draw here anyway"
            # override, so a box can still be drawn inside/over another one.
            force_new = bool(event.modifiers() & Qt.ShiftModifier)
            if (clicked_empty or force_new) and self._pixmap_item.boundingRect().contains(scene_pos):
                self._draw_start = scene_pos
                pen = QPen(QColor(_DRAW_PEN_COLOR))
                pen.setStyle(Qt.DashLine)
                pen.setWidth(2)
                self._draw_item = self._scene.addRect(QRectF(scene_pos, scene_pos), pen)
                self.status_changed.emit("Drawing… release the mouse to finish the box.")
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._draw_item is not None and self._draw_start is not None:
            rect = QRectF(self._draw_start, self.mapToScene(event.pos())).normalized()
            if self._pixmap_item is not None:
                rect = rect.intersected(self._pixmap_item.boundingRect())
            self._draw_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._draw_item is not None:
            rect = self._draw_item.rect()
            self._draw_start = None
            if rect.width() >= MIN_DRAW_SIZE and rect.height() >= MIN_DRAW_SIZE:
                # Keep the dashed rect on screen (as a placeholder, not yet
                # a real annotation) while the class picker is open, so the
                # box the user just drew doesn't visually vanish mid-pick.
                self.status_changed.emit("Pick a class for this box…")
                self._prompt_class_for_new_box(rect)
            else:
                self._scene.removeItem(self._draw_item)
                self._draw_item = None
                self.refresh_status()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _prompt_class_for_new_box(self, rect: QRectF) -> None:
        project = self._controller.project
        if project is None or not project.classes or self._image_id is None:
            self._cancel_pending_box()
            return
        anchor = self.mapToGlobal(self.mapFromScene(rect.topLeft()))
        popup = ClassPickerPopup(project.classes, self)
        self._pending_picker = popup
        popup.class_picked.connect(lambda class_id, r=rect: self._commit_new_box(r, class_id))
        popup.destroyed.connect(self._on_picker_closed)
        popup.popup_at(anchor)

    def _on_picker_closed(self) -> None:
        # Catches every dismissal path (Escape, click-outside, a class
        # actually picked) — if the drawn rect is still sitting there
        # un-persisted at this point, the popup closed without a pick, so
        # discard it rather than leaving an unclassed box behind.
        self._pending_picker = None
        if self._draw_item is not None:
            self._scene.removeItem(self._draw_item)
            self._draw_item = None
        self.refresh_status()

    def _cancel_pending_box(self) -> None:
        if self._pending_picker is not None:
            self._pending_picker.close()
        elif self._draw_item is not None:
            self._scene.removeItem(self._draw_item)
            self._draw_item = None

    def _commit_new_box(self, rect: QRectF, class_id: str) -> None:
        if self._draw_item is not None:
            self._scene.removeItem(self._draw_item)
            self._draw_item = None
        if not self._image_id:
            return
        annotation = self._controller.add_annotation(self._image_id, class_id, self._rect_to_points(rect))
        item = self._add_box_item(annotation.id, class_id, rect)
        self._scene.clearSelection()
        item.setSelected(True)
        self.refresh_status()

    # -- editing/deleting existing boxes -----------------------------------

    def _on_box_edit_requested(self, item: BoxItem) -> None:
        project = self._controller.project
        if project is None or not project.classes:
            return
        anchor = self.mapToGlobal(self.mapFromScene(item.rect().topLeft()))
        popup = ClassPickerPopup(project.classes, self)
        annotation_id = item.annotation_id
        popup.class_picked.connect(lambda class_id, aid=annotation_id: self._reclassify_box(aid, class_id))
        popup.popup_at(anchor)

    def _reclassify_box(self, annotation_id: str | None, class_id: str) -> None:
        if not annotation_id or not self._image_id:
            return
        self._controller.update_annotation(self._image_id, annotation_id, class_id=class_id)
        self.update_item_class(annotation_id, class_id)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_selected()
            return
        super().keyPressEvent(event)

    def _delete_selected(self) -> None:
        selected = [item for item in self._scene.selectedItems() if isinstance(item, BoxItem)]
        for item in selected:
            self._delete_item(item)

    def _delete_item(self, item: BoxItem) -> None:
        annotation_id = item.annotation_id
        self._scene.removeItem(item)
        if annotation_id:
            self._boxes.pop(annotation_id, None)
            self._controller.remove_annotation(self._image_id, annotation_id)
