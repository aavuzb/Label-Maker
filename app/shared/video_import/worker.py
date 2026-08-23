"""Extracts frames from a video file at a target sampling rate, off the UI
thread. Pure OpenCV decode/write — nothing here touches the ProjectController
or any Qt widget, since neither is thread-safe. Results travel out as
signals; the dialog that owns this worker imports the written frames on the
main thread.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from app.shared.autolabel.cpu_budget import cpu_thread_budget

_JPEG_QUALITY = 95
# Frames between scheduler yields — frequent enough to keep the UI responsive
# during a long decode loop without adding meaningful overhead per frame.
_YIELD_EVERY = 15


class VideoImportWorker(QThread):
    progress = Signal(int, int)  # frames scanned so far, total hint (0 if unknown)
    frame_saved = Signal(str)  # path of one written frame
    finished_all = Signal(int, int, bool, str)  # saved_count, scanned_count, cancelled, error ("" if none)

    def __init__(self, video_path: Path, output_dir: Path, target_fps: float) -> None:
        super().__init__()
        self._video_path = video_path
        self._output_dir = output_dir
        self._target_fps = target_fps
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        cv2.setNumThreads(cpu_thread_budget())
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            cap.release()
            self.finished_all.emit(0, 0, False, f"Could not open '{self._video_path.name}' as a video file.")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Unknown/broken FPS metadata: fall back to keeping every decoded
        # frame rather than guessing a rate that could wildly over- or
        # under-sample.
        interval = (video_fps / self._target_fps) if video_fps > 0 else 1.0
        interval = max(interval, 1e-6)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        pad_width = max(5, len(str(total_hint)))

        scanned = 0
        saved = 0
        next_capture_at = 0.0
        error = ""
        try:
            while not self._stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break
                if scanned >= next_capture_at:
                    out_path = self._output_dir / f"frame_{saved + 1:0{pad_width}d}.jpg"
                    if cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]):
                        saved += 1
                        self.frame_saved.emit(str(out_path))
                    next_capture_at += interval
                scanned += 1
                self.progress.emit(scanned, total_hint)
                if scanned % _YIELD_EVERY == 0:
                    self.msleep(1)
        except Exception as exc:  # noqa: BLE001 — surface any decode error, keep whatever was already saved
            error = str(exc)
        finally:
            cap.release()

        self.finished_all.emit(saved, scanned, self._stop_requested, error)
