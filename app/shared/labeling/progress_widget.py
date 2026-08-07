"""Bottom bar summarizing labeling progress for the open project."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QWidget

from app.features.project.manager import ProjectController
from app.shared.theme import BG_SURFACE, BORDER, FS_CAPTION, FS_H1, TEXT_MUTED, TEXT_PRIMARY

_LABELED_COLOR = "#10b981"  # matches the app's general "complete" green


def _stat_label() -> QLabel:
    label = QLabel()
    label.setTextFormat(Qt.RichText)
    label.setStyleSheet("border: none;")
    return label


def _set_stat(label: QLabel, value: int, caption: str, color: str) -> None:
    label.setText(
        f'<span style="font-size:{FS_H1 - 4}px; font-weight:800; color:{color};">{value}</span> '
        f'<span style="font-size:{FS_CAPTION}px; font-weight:600; color:{TEXT_MUTED};">{caption}</span>'
    )


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setStyleSheet(f"color: {BORDER};")
    line.setFixedHeight(16)
    return line


class ProgressWidget(QWidget):
    def __init__(self, controller: ProjectController) -> None:
        super().__init__()
        self._controller = controller
        self._refresh_pending = False
        self.setStyleSheet(f"ProgressWidget {{ background-color: {BG_SURFACE}; border-top: 1px solid {BORDER}; }}")

        self.total_label = _stat_label()
        self.labeled_label = _stat_label()
        self.remaining_label = _stat_label()
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setTextVisible(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(12)
        layout.addWidget(self.total_label)
        layout.addWidget(_divider())
        layout.addWidget(self.labeled_label)
        layout.addWidget(_divider())
        layout.addWidget(self.remaining_label)
        layout.addStretch()
        layout.addWidget(self.progress_bar)

        controller.project_opened.connect(self.refresh)
        controller.images_added.connect(lambda _added: self._schedule_refresh())
        controller.images_removed.connect(lambda _removed: self._schedule_refresh())
        controller.image_updated.connect(lambda _id: self._schedule_refresh())
        self.refresh()

    def _schedule_refresh(self) -> None:
        """Coalesces a burst of signals (e.g. hundreds of predictions
        applied back-to-back during auto-labeling) into a single refresh()
        instead of one full images/labeled/remaining recount per signal."""
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self._run_scheduled_refresh)

    def _run_scheduled_refresh(self) -> None:
        self._refresh_pending = False
        self.refresh()

    def refresh(self) -> None:
        project = self._controller.project
        if project is None:
            _set_stat(self.total_label, 0, "Total", TEXT_PRIMARY)
            _set_stat(self.labeled_label, 0, "Labeled", _LABELED_COLOR)
            _set_stat(self.remaining_label, 0, "Remaining", TEXT_PRIMARY)
            self.progress_bar.setValue(0)
            return

        total = len(project.images)
        # entry.state covers both paradigms: classification tags class_id
        # directly, detection/segmentation tag via annotations instead.
        labeled = sum(1 for i in project.images if i.state == "assigned")
        remaining = total - labeled
        percent = int(round(100 * labeled / total)) if total else 0

        _set_stat(self.total_label, total, "Total", TEXT_PRIMARY)
        _set_stat(self.labeled_label, labeled, "Labeled", _LABELED_COLOR)
        _set_stat(self.remaining_label, remaining, "Remaining", TEXT_PRIMARY)
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{percent}%")
