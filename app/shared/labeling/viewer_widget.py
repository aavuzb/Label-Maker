"""Single large image preview (classification) — no editing, just viewing."""

from __future__ import annotations

from PySide6.QtWidgets import QGraphicsView, QLabel

from app.shared.labeling.zoomable_view import ZoomableGraphicsView
from app.shared.theme import FS_BODY, readable_text_color

_BADGE_MARGIN = 12


class ImageViewerWidget(ZoomableGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        # No drawing interaction here, so click-drag can freely pan the view.
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        # A fixed-size overlay (child of the viewport, not the graphics
        # scene) so it reads at a constant, legible size regardless of the
        # image's own zoom level — matching how the image grid's own class
        # chip (see delegate.py) is a fixed part of the thumbnail cell, not
        # something that shrinks/grows with the image.
        self._class_badge = QLabel(self.viewport())
        self._class_badge.hide()

    def set_image_path(self, path: str | None) -> None:
        self._load_pixmap(path)

    def set_class_label(self, name: str | None, color: str | None) -> None:
        """Shows a colored chip naming the current image's assigned class at
        the top of the preview — pass None to hide it (unlabeled image, or
        no image loaded at all)."""
        if not name:
            self._class_badge.hide()
            return
        bg = color or "#3498db"
        text_color = readable_text_color(bg).name()
        self._class_badge.setText(name)
        self._class_badge.setStyleSheet(
            f"""
            QLabel {{
                background-color: {bg};
                color: {text_color};
                font-weight: 700;
                font-size: {FS_BODY}px;
                padding: 5px 14px;
                border-radius: 6px;
            }}
            """
        )
        self._class_badge.adjustSize()
        self._position_badge()
        self._class_badge.show()
        self._class_badge.raise_()

    def _position_badge(self) -> None:
        """Centered on the image's own on-screen top edge, not the viewport's
        — fit-to-screen letterboxes any image whose aspect ratio doesn't
        match the panel, so anchoring to the viewport alone would float the
        badge over blank space above a narrow/small image instead of
        actually sitting on top of it."""
        if self._pixmap_item is not None:
            top_left = self.mapFromScene(self._pixmap_item.sceneBoundingRect().topLeft())
            top_right = self.mapFromScene(self._pixmap_item.sceneBoundingRect().topRight())
            center_x = (top_left.x() + top_right.x()) // 2
            top_y = min(top_left.y(), top_right.y())
        else:
            center_x = self.viewport().width() // 2
            top_y = 0
        x = center_x - self._class_badge.width() // 2
        y = top_y + _BADGE_MARGIN
        self._class_badge.move(max(0, x), max(0, y))

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        if self._class_badge.isVisible():
            self._position_badge()

    def zoom_in(self) -> None:
        super().zoom_in()
        if self._class_badge.isVisible():
            self._position_badge()

    def zoom_out(self) -> None:
        super().zoom_out()
        if self._class_badge.isVisible():
            self._position_badge()

    def fit_to_screen(self) -> None:
        super().fit_to_screen()
        if self._class_badge.isVisible():
            self._position_badge()
