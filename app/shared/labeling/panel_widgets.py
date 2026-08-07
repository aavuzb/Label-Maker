"""Small shared UI building blocks used across the three task workspaces."""

from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QPushButton, QWidget

from app.shared.theme import ACCENT, FS_CAPTION, FS_H1, FS_H2, FS_SMALL, TEXT_MUTED, TEXT_PRIMARY, tint_rgba

_ZOOM_BUTTON_STYLE = f"""
    QPushButton {{ min-width: 29px; max-width: 29px; min-height: 29px; max-height: 29px;
                  padding: 0px; font-weight: 700; font-size: {FS_H1 - 4}px; }}
"""


def card_frame() -> QFrame:
    """A soft-elevated surface (rounded, bordered, subtly shadowed) used as
    the container for every panel/column, so each section of a workspace
    reads as a distinct card floating on the app's gray background rather
    than a flat, borderless region."""
    frame = QFrame()
    frame.setObjectName("Card")  # styled via APP_STYLESHEET's QFrame#Card rule
    effect = QGraphicsDropShadowEffect(frame)
    effect.setBlurRadius(16)
    effect.setXOffset(0)
    effect.setYOffset(2)
    effect.setColor(QColor(15, 23, 42, 22))
    frame.setGraphicsEffect(effect)
    return frame


def panel_header(text: str, subtitle: str = "", accent: str = "") -> QWidget:
    """A bold uppercase panel title, optionally with a smaller muted subtitle
    line underneath, optionally prefixed with a small color chip tying the
    panel to its task's accent color."""
    html = f'<span style="font-size:{FS_H2}px; font-weight:700; letter-spacing:1px;">{text}</span>'
    if subtitle:
        html += f'<br><span style="font-size:{FS_CAPTION}px; font-weight:400; color:{TEXT_MUTED};">{subtitle}</span>'
    label = QLabel(html)
    label.setWordWrap(True)

    if not accent:
        label.setStyleSheet(f"color: {TEXT_MUTED}; padding: 2px 4px 6px 4px; border: none;")
        return label

    label.setStyleSheet(f"color: {TEXT_MUTED}; padding: 0px; border: none;")
    chip = QLabel()
    chip.setFixedSize(4, 20 if subtitle else 14)
    chip.setStyleSheet(f"background-color: {accent}; border-radius: 2px; border: none;")

    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(2, 2, 4, 8)
    row_layout.setSpacing(9)
    row_layout.addWidget(chip)
    row_layout.addWidget(label, stretch=1)
    return row


def build_callout(text: str, accent: str = ACCENT) -> QLabel:
    """A tinted, left-accented info banner — for short how-to instructions.
    Plain-text QLabels honor embedded newlines as line breaks, so callers
    can pass multi-line/numbered step text directly."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(
        f"""
        QLabel {{
            background-color: {tint_rgba(accent, 0.08)};
            border: none;
            border-left: 3px solid {accent};
            border-radius: 4px;
            padding: 10px 12px;
            color: {TEXT_PRIMARY};
            font-size: {FS_SMALL}px;
            line-height: 150%;
        }}
        """
    )
    return label


def status_pill(accent: str = ACCENT) -> QLabel:
    """A small, muted single-line hint strip — for live "what's happening
    right now" feedback (e.g. "Point 3 placed — click near the first point
    to close"). Callers just call .setText() on the result to update it."""
    label = QLabel("")
    label.setWordWrap(True)
    label.setStyleSheet(
        f"""
        QLabel {{
            color: {accent};
            font-size: {FS_CAPTION}px;
            font-weight: 600;
            padding: 0px 2px;
            border: none;
        }}
        """
    )
    return label


def build_zoom_row(zoom_out: Callable[[], None], zoom_in: Callable[[], None], fit: Callable[[], None] | None = None) -> QHBoxLayout:
    out_button = QPushButton("−")
    out_button.setStyleSheet(_ZOOM_BUTTON_STYLE)
    out_button.setToolTip("Zoom out")
    out_button.clicked.connect(zoom_out)

    in_button = QPushButton("+")
    in_button.setStyleSheet(_ZOOM_BUTTON_STYLE)
    in_button.setToolTip("Zoom in")
    in_button.clicked.connect(zoom_in)

    row = QHBoxLayout()
    row.setSpacing(4)
    if fit is not None:
        row.addStretch()
    row.addWidget(out_button)
    if fit is not None:
        fit_button = QPushButton("Fit")
        fit_button.setStyleSheet(f"padding: 5px 12px; font-size: {FS_SMALL}px;")
        fit_button.setToolTip("Fit image to view (F)")
        fit_button.clicked.connect(fit)
        row.addWidget(fit_button)
    row.addWidget(in_button)
    if fit is not None:
        row.addStretch()
    return row
