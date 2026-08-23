"""Dialog for importing video frames into the project as images — browse a
video, choose a sampling rate, and extract frames on a background thread."""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from app.features.project.manager import ProjectController
from app.shared.ids import new_id
from app.shared.labeling.panel_widgets import build_callout
from app.shared.paths import safe_folder_name
from app.shared.theme import ACCENT, FS_CAPTION, TEXT_MUTED, primary_button_style
from app.shared.video_import.worker import VideoImportWorker

_VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.mpg *.mpeg);;All Files (*)"
_ERROR_COLOR = "#dc2626"


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class VideoImportDialog(QDialog):
    def __init__(self, controller: ProjectController, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._worker: VideoImportWorker | None = None
        self._frame_paths: list[str] = []
        self._video_path: Path | None = None
        self._video_fps = 0.0
        self._video_frame_count = 0
        self._video_duration: float | None = None

        self.setWindowTitle("Import Video")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addWidget(
            build_callout(
                "Pick a video and a sampling rate — frames are extracted and added to this "
                "project as images, ready to label.",
                ACCENT,
            )
        )

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Choose a video file…")
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_video)
        self._browse_button = browse_button
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(browse_button)

        self.video_info_label = QLabel("")
        self.video_info_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FS_CAPTION}px;")

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 60.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.setValue(10.0)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self._update_estimate)

        self.estimate_label = QLabel("")
        self.estimate_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FS_CAPTION}px;")

        form = QFormLayout()
        form.setSpacing(10)
        form.addRow("Video file", path_row)
        form.addRow("", self.video_info_label)
        form.addRow("Frame rate", self.fps_spin)
        form.addRow("", self.estimate_label)
        layout.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.upload_button = self.buttons.button(QDialogButtonBox.Ok)
        self.upload_button.setText("Upload")
        self.upload_button.setStyleSheet(primary_button_style(ACCENT))
        self.upload_button.setEnabled(False)
        self.cancel_button = self.buttons.button(QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._start_import)
        self.buttons.rejected.connect(self._on_cancel_clicked)
        layout.addWidget(self.buttons)

    # -- video selection ------------------------------------------------

    def _browse_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", _VIDEO_FILTER)
        if not file_path:
            return

        path = Path(file_path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            QMessageBox.warning(self, "Import Video", f"Could not open '{path.name}' as a video file.")
            return

        self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        self._video_path = path
        self._video_duration = (
            self._video_frame_count / self._video_fps
            if self._video_fps > 0 and self._video_frame_count > 0
            else None
        )

        self.path_edit.setText(str(path))
        info_parts = []
        if width and height:
            info_parts.append(f"{width}×{height}")
        if self._video_duration is not None:
            info_parts.append(_format_duration(self._video_duration))
        if self._video_fps > 0:
            info_parts.append(f"source ~{self._video_fps:.1f} fps")
        self.video_info_label.setText(" • ".join(info_parts) if info_parts else "Video selected")

        self.upload_button.setEnabled(True)
        self._update_estimate()

    def _update_estimate(self) -> None:
        if self._video_path is None:
            self.estimate_label.setText("")
            return
        target_fps = self.fps_spin.value()
        if self._video_duration is not None:
            estimated = max(1, round(self._video_duration * target_fps))
            if self._video_frame_count > 0:
                estimated = min(estimated, self._video_frame_count)
            self.estimate_label.setText(f"≈ {estimated} frame(s) will be extracted")
        else:
            self.estimate_label.setText("Frame count unknown — will extract every frame")

    # -- import run -------------------------------------------------------

    def _start_import(self) -> None:
        project = self._controller.project
        if project is None or self._video_path is None:
            return
        if not self._video_path.exists():
            QMessageBox.warning(self, "Import Video", "The selected video file no longer exists.")
            return

        output_dir = project.project_dir / "video_frames" / f"{safe_folder_name(self._video_path.stem)}_{new_id()}"

        self._frame_paths = []
        self._set_form_enabled(False)
        self.progress_bar.setRange(0, self._video_frame_count if self._video_frame_count > 0 else 0)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Extracting frames…")
        self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {FS_CAPTION}px;")
        self.status_label.setVisible(True)
        self.upload_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self._worker = VideoImportWorker(self._video_path, output_dir, self.fps_spin.value())
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_saved.connect(self._frame_paths.append)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.start(QThread.LowPriority)

    def _on_progress(self, scanned: int, _total: int) -> None:
        self.progress_bar.setValue(scanned)

    def _on_finished(self, saved: int, _scanned: int, cancelled: bool, error: str) -> None:
        self._worker = None

        if error:
            self._set_form_enabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setVisible(False)
            QMessageBox.critical(self, "Import Video", f"Could not read this video: {error}")
            return

        added = self._controller.import_paths([Path(p) for p in self._frame_paths]) if self._frame_paths else 0

        if added == 0:
            self._set_form_enabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setVisible(False)
            QMessageBox.warning(self, "Import Video", "No frames were extracted.")
            return

        summary = f"Imported {added} frame(s) from {self._video_path.name}."
        if cancelled:
            summary += " (Extraction was cancelled early — frames captured so far were still imported.)"
        QMessageBox.information(self, "Import Video", summary)
        self.accept()

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Stopping…")
        else:
            self.reject()

    def _set_form_enabled(self, enabled: bool) -> None:
        self._browse_button.setEnabled(enabled)
        self.fps_spin.setEnabled(enabled)
        self.upload_button.setEnabled(enabled and self._video_path is not None)

    # -- lifecycle ----------------------------------------------------------

    def reject(self) -> None:  # noqa: N802 — Qt override
        self._stop_and_wait()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override
        self._stop_and_wait()
        super().closeEvent(event)

    def _stop_and_wait(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(5000)
