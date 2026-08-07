"""Runs auto-label inference off the UI thread.

Only pure computation happens inside `run()` — model inference and
read-only image file access — nothing here touches the ProjectController
or any Qt widget, since neither is thread-safe. Results travel out as
signals; the dialog that owns this worker applies them to the project on
the main thread.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QThread, Signal

from app.shared.autolabel.prediction import Prediction


class AutoLabelWorker(QThread):
    progress = Signal(int, int)  # images processed so far, total
    image_labeled = Signal(str, object)  # image_id, list[Prediction] (non-empty)
    image_failed = Signal(str, str)  # image_id, error message
    finished_all = Signal(int, int)  # images_labeled, images_processed (may be < total if stopped)

    def __init__(
        self,
        image_ids: list[str],
        path_by_id: dict[str, str],
        predict_fn: Callable[[str], list[Prediction]],
    ) -> None:
        super().__init__()
        self._image_ids = image_ids
        self._path_by_id = path_by_id
        self._predict_fn = predict_fn
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        labeled_count = 0
        processed_count = 0
        total = len(self._image_ids)
        for image_id in self._image_ids:
            if self._stop_requested:
                break
            path = self._path_by_id.get(image_id)
            try:
                predictions = self._predict_fn(path) if path else []
            except Exception as exc:  # noqa: BLE001 — surface any decode/inference error, keep the batch going
                processed_count += 1
                self.image_failed.emit(image_id, str(exc))
                self.progress.emit(processed_count, total)
                continue
            processed_count += 1
            if predictions:
                labeled_count += 1
                self.image_labeled.emit(image_id, predictions)
            self.progress.emit(processed_count, total)
            # A deliberate, tiny yield after every image. Model inference
            # already runs off the UI thread, but back-to-back GPU/CPU work
            # across a whole dataset can still starve the rest of the
            # system of scheduling opportunities on a machine with few
            # cores — this costs a fraction of a second over an entire run
            # and keeps the OS able to interleave the UI (and everything
            # else) in between images instead of getting shut out for
            # minutes at a time.
            self.msleep(1)
        self.finished_all.emit(labeled_count, processed_count)
