"""Paints each thumbnail cell with a class-color border."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from app.shared.labeling.image_model import AnnotationCountRole, ClassColorRole, ClassNameRole
from app.shared.theme import readable_text_color

CELL_MARGIN = 6
BORDER_WIDTH = 3
ANNOTATED_COLOR = "#10b981"
CLASS_LABEL_HEIGHT = 24


class ThumbnailDelegate(QStyledItemDelegate):
    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:  # noqa: N802
        base = super().sizeHint(option, index)
        return QSize(base.width(), base.height() + 8)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # noqa: N802
        painter.save()

        rect = option.rect.adjusted(CELL_MARGIN, CELL_MARGIN, -CELL_MARGIN, -CELL_MARGIN)

        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        color_hex = index.data(ClassColorRole)
        annotation_count = index.data(AnnotationCountRole) or 0
        if color_hex or annotation_count:
            pen = QPen(QColor(color_hex or ANNOTATED_COLOR))
            pen.setWidth(BORDER_WIDTH)
            painter.setPen(pen)
            painter.drawRect(QRectF(rect).adjusted(1, 1, -1, -1))

        # A color-only border doesn't scale once a project has many classes
        # (or classes with similar hues) — a name is unambiguous, a color
        # isn't. Reserve room at the top of the thumbnail for it.
        class_name = index.data(ClassNameRole)
        top_reserved = CLASS_LABEL_HEIGHT if class_name else 0

        thumbnail = index.data(Qt.DecorationRole)
        if thumbnail is not None:
            pixmap = thumbnail.scaled(
                rect.width() - 2 * BORDER_WIDTH,
                rect.height() - 2 * BORDER_WIDTH - 16 - top_reserved,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            target = rect.adjusted(BORDER_WIDTH, BORDER_WIDTH + top_reserved, -BORDER_WIDTH, -BORDER_WIDTH - 16)
            x = target.x() + (target.width() - pixmap.width()) // 2
            y = target.y() + (target.height() - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)

        if class_name:
            label_rect = QRectF(
                rect.left() + BORDER_WIDTH + 2,
                rect.top() + BORDER_WIDTH + 2,
                rect.width() - 2 * (BORDER_WIDTH + 2),
                top_reserved - 4,
            )
            chip_color = QColor(color_hex or ANNOTATED_COLOR)
            painter.setPen(Qt.NoPen)
            painter.setBrush(chip_color)
            painter.drawRoundedRect(label_rect, 4, 4)
            font = QFont(painter.font())
            font.setBold(True)
            # `+2` alone would go invisible if the base font's size happens
            # to be reported in pixels (pointSize() == -1 in that case) —
            # always floor to a size that's actually legible.
            font.setPointSize(max(12, font.pointSize() + 2))
            painter.setFont(font)
            painter.setPen(QPen(readable_text_color(chip_color.name())))
            elided = painter.fontMetrics().elidedText(class_name, Qt.ElideRight, int(label_rect.width()) - 6)
            painter.drawText(label_rect, Qt.AlignCenter, elided)

        painter.setFont(option.font)
        text = index.data(Qt.DisplayRole) or ""
        text_rect = rect.adjusted(2, rect.height() - 16, -2, -2)
        painter.setPen(QPen(option.palette.text().color()))
        elided = painter.fontMetrics().elidedText(text, Qt.ElideMiddle, text_rect.width())
        painter.drawText(text_rect, Qt.AlignCenter, elided)

        if annotation_count:
            badge_text = str(annotation_count)
            font = QFont(painter.font())
            font.setBold(True)
            # Floored rather than shrunk (was `-1`): a bare `+1` alone would
            # go invisible if the base font's size happens to be reported
            # in pixels (pointSize() == -1 in that case) — see the same
            # fix on the class-name chip above.
            font.setPointSize(max(10, font.pointSize() + 1))
            painter.setFont(font)
            metrics = painter.fontMetrics()
            badge_w = max(22, metrics.horizontalAdvance(badge_text) + 12)
            badge_h = 20
            badge_rect = QRectF(rect.right() - badge_w - 4, rect.top() + 4, badge_w, badge_h)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(ANNOTATED_COLOR))
            painter.drawRoundedRect(badge_rect, 9, 9)
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(badge_rect, Qt.AlignCenter, badge_text)

        painter.restore()
