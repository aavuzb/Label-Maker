"""Landing page shown on launch: intro text + task selection."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.features.project.models import TASK_CLASSIFICATION, TASK_DETECTION, TASK_SEGMENTATION
from app.shared.theme import (
    ACCENT,
    BG_MAIN,
    BG_SURFACE,
    BORDER,
    FS_BODY,
    FS_DISPLAY,
    FS_H1,
    RADIUS_CARD,
    TASK_ACCENTS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    app_icon_pixmap,
    tint_rgba,
)

_TASKS = [
    (TASK_CLASSIFICATION, "Classification", "Tag whole images with a single class."),
    (TASK_DETECTION, "Detection", "Draw bounding boxes around objects."),
    (TASK_SEGMENTATION, "Segmentation", "Trace polygon masks around objects."),
]

_CARD_ICON_SIZE = 46


def _add_elevation(widget: QWidget, blur: int = 24, y_offset: int = 6, alpha: int = 40) -> None:
    """A soft drop shadow so a surface reads as raised/tappable rather than
    flat-printed on the background."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setXOffset(0)
    shadow.setYOffset(y_offset)
    shadow.setColor(QColor(15, 23, 42, alpha))
    widget.setGraphicsEffect(shadow)


def _task_icon_pixmap(accent: str, kind: str, size: int = _CARD_ICON_SIZE) -> QPixmap:
    """A small glyph distinguishing each task at a glance, drawn directly
    (no icon-font/SVG asset dependency) on a soft tinted badge in the task's
    own accent color — the same color already used for that task's class
    chips, progress bar, and buttons throughout the app."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    color = QColor(accent)
    badge_color = QColor(accent)
    badge_color.setAlphaF(0.14)  # QColor's string constructor can't parse tint_rgba()'s "rgba(...)" CSS syntax
    painter.setPen(Qt.NoPen)
    painter.setBrush(badge_color)
    painter.drawEllipse(0, 0, size, size)

    pen = QPen(color)
    pen.setWidthF(size * 0.09)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    m = size * 0.28  # inner margin common to all three glyphs

    if kind == TASK_CLASSIFICATION:
        # A checkmark — "this image has been tagged."
        path = QPainterPath()
        path.moveTo(m, size * 0.52)
        path.lineTo(size * 0.44, size * 0.68)
        path.lineTo(size - m, size * 0.34)
        painter.drawPath(path)
    elif kind == TASK_DETECTION:
        # Four corner brackets — a camera/viewfinder framing an object.
        arm = size * 0.16
        corners = [(m, m, 1, 1), (size - m, m, -1, 1), (m, size - m, 1, -1), (size - m, size - m, -1, -1)]
        for x, y, dx, dy in corners:
            painter.drawLine(QPointF(x, y), QPointF(x + arm * dx, y))
            painter.drawLine(QPointF(x, y), QPointF(x, y + arm * dy))
    else:
        # An irregular closed polygon with vertex dots — tracing a mask outline.
        points = [
            QPointF(size * 0.30, size * 0.26),
            QPointF(size * 0.72, size * 0.22),
            QPointF(size * 0.78, size * 0.58),
            QPointF(size * 0.52, size * 0.80),
            QPointF(size * 0.22, size * 0.62),
        ]
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        path.closeSubpath()
        painter.drawPath(path)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        dot_r = size * 0.05
        for point in points:
            painter.drawEllipse(point, dot_r, dot_r)

    painter.end()
    return pixmap


class _TaskCard(QFrame):
    """A clickable card — icon, title, description — for choosing a task on
    the landing page. A plain QWidget (not QPushButton) so the title and
    description can carry different type styling within one card; click
    handling is done directly rather than via a button, and every child
    label is made click-through so pressing anywhere on the card (including
    directly over the text) fires it, not just the padding around it."""

    clicked = Signal()

    def __init__(self, task_type: str, accent: str, name: str, description: str) -> None:
        super().__init__()
        self._accent = accent
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover, True)
        self.setMinimumSize(216, 168)
        self._apply_style(hovered=False)

        # Each label gets an explicit "background: transparent" — a widget
        # with its own local styleSheet (this card's, set in _apply_style)
        # makes Qt paint child QLabels with an opaque palette-based
        # background by default instead of letting the card's own
        # background show through, unless told otherwise.
        icon_label = QLabel()
        icon_label.setPixmap(_task_icon_pixmap(accent, task_type))
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("background: transparent; border: none;")

        title_label = QLabel(name)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"background: transparent; font-size: {FS_H1}px; font-weight: 700; color: {TEXT_PRIMARY}; border: none;")

        desc_label = QLabel(description)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"background: transparent; font-size: {FS_BODY}px; color: {TEXT_MUTED}; border: none;")

        for child in (icon_label, title_label, desc_label):
            child.setAttribute(Qt.WA_TransparentForMouseEvents)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 24, 20, 22)
        layout.setSpacing(10)
        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch(1)

        _add_elevation(self)

    def _apply_style(self, hovered: bool) -> None:
        border_color = self._accent if hovered else BORDER
        bg = tint_rgba(self._accent, 0.07) if hovered else BG_SURFACE
        self.setStyleSheet(
            f"""
            _TaskCard {{
                background-color: {bg};
                border: 1.5px solid {border_color};
                border-top: 4px solid {self._accent};
                border-radius: {RADIUS_CARD}px;
            }}
            """
        )

    def enterEvent(self, event) -> None:  # noqa: N802 — Qt override
        self._apply_style(hovered=True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 — Qt override
        self._apply_style(hovered=False)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt override
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class LandingPage(QWidget):
    task_selected = Signal(str)
    open_project_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet(
            f"LandingPage {{ background: qradialgradient(cx:0.5, cy:0.32, radius:0.9, "
            f"fx:0.5, fy:0.32, stop:0 {tint_rgba(ACCENT, 0.05)}, stop:1 {BG_MAIN}); }}"
        )

        logo_label = QLabel()
        logo_label.setPixmap(app_icon_pixmap(56))

        title = QLabel("LabelMaker")
        title.setStyleSheet(f"font-size: {FS_DISPLAY}px; font-weight: 800; letter-spacing: -0.5px; color: {TEXT_PRIMARY}; border: none;")

        logo_row = QHBoxLayout()
        logo_row.setSpacing(14)
        logo_row.addStretch()
        logo_row.addWidget(logo_label)
        logo_row.addWidget(title)
        logo_row.addStretch()

        intro = QLabel(
            "A desktop dataset preparation tool for computer vision. Import images, define "
            "classes, and label them for classification, object detection, or semantic "
            "segmentation — your original files are never moved."
        )
        intro.setAlignment(Qt.AlignCenter)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FS_BODY}px; line-height: 150%; border: none;")

        prompt = QLabel("What would you like to do?")
        prompt.setAlignment(Qt.AlignCenter)
        prompt.setStyleSheet(f"font-size: {FS_H1 - 4}px; font-weight: 700; margin-top: 8px; color: {TEXT_PRIMARY}; border: none;")

        button_row = QHBoxLayout()
        button_row.setSpacing(18)
        for task_type, name, description in _TASKS:
            accent = TASK_ACCENTS[task_type]
            card = _TaskCard(task_type, accent, name, description)
            card.clicked.connect(lambda t=task_type: self.task_selected.emit(t))
            button_row.addWidget(card)

        open_button = QPushButton("Open Existing Project…")
        open_button.setCursor(Qt.PointingHandCursor)
        open_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                color: {TEXT_MUTED};
                font-weight: 600;
                font-size: {FS_BODY}px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{ color: {ACCENT}; text-decoration: underline; }}
            """
        )
        open_button.clicked.connect(self.open_project_requested.emit)

        text_block = QWidget()
        text_block.setMaximumWidth(600)
        text_layout = QVBoxLayout(text_block)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(10)
        text_layout.addLayout(logo_row)
        text_layout.addWidget(intro)

        text_row = QHBoxLayout()
        text_row.addStretch()
        text_row.addWidget(text_block)
        text_row.addStretch()

        center = QVBoxLayout()
        center.addLayout(text_row)
        center.addWidget(prompt)
        center.addLayout(button_row)
        center.addWidget(open_button, alignment=Qt.AlignHCenter)
        center.setSpacing(18)

        outer = QVBoxLayout(self)
        outer.addStretch(3)
        outer.addLayout(center)
        outer.addStretch(4)
