"""Auto-labeling for Detection: image -> bounding boxes.

Expected model contract: a single image input tensor (NCHW, square), and a
single output tensor holding one row per candidate box of either:
  - 4 + num_classes values: box(cx, cy, w, h) + per-class scores
    (Ultralytics YOLOv8/v11 ONNX export layout — no separate objectness).
  - 5 + num_classes values: box(cx, cy, w, h) + objectness + per-class
    scores (YOLOv5-style layout; final confidence = objectness * class score).
Box coordinates are in the model's own (letterboxed) input pixel space.
Both "channels-first" (C, N) and "boxes-first" (N, C) output orientations
are auto-detected from which axis matches 4/5 + num_classes.

Raw Ultralytics YOLO checkpoints (ModelInfo.format == FORMAT_ULTRALYTICS,
see model.py) don't go through any of the above — their raw output layout
isn't consistent across YOLO versions (v10/v26-style heads already return
decoded, post-NMS boxes rather than per-class scores), so `_predict_ultralytics`
routes those through Ultralytics' own `.predict()` instead.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from app.features.project.manager import ProjectController
from app.shared.autolabel.model import FORMAT_ULTRALYTICS, ModelInfo
from app.shared.autolabel.preprocess import bgr_to_tensor, letterbox, read_image_bgr
from app.shared.autolabel.prediction import Prediction

DEFAULT_SIZE = 640


def _predict_ultralytics(
    model: ModelInfo,
    image: np.ndarray,
    class_mapping: dict[int, str],
    confidence_threshold: float,
    iou_threshold: float,
) -> list[Prediction]:
    """Raw-output layout varies across YOLO versions (v8/v11 emit per-class
    scores that need NMS here; v10/v26-style heads already return decoded,
    post-NMS boxes) — routing through Ultralytics' own `.predict()` sidesteps
    having to detect and replicate each version's convention by hand."""
    yolo = model.backend.yolo
    orig_h, orig_w = image.shape[:2]
    results = yolo.predict(image, conf=confidence_threshold, iou=iou_threshold, device=model.backend.device_str, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    predictions: list[Prediction] = []
    for xyxy, confidence, class_index in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
    ):
        class_id = class_mapping.get(int(class_index))
        if class_id is None:
            continue
        x1, y1, x2, y2 = xyxy
        x1, x2 = sorted((max(0.0, min(x1, orig_w)), max(0.0, min(x2, orig_w))))
        y1, y2 = sorted((max(0.0, min(y1, orig_h)), max(0.0, min(y2, orig_h))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        predictions.append(
            Prediction(
                class_id=class_id,
                confidence=float(confidence),
                points=[(x1 / orig_w, y1 / orig_h), (x2 / orig_w, y2 / orig_h)],
            )
        )
    return predictions


def _normalize_rows(raw: np.ndarray, num_classes: int) -> tuple[np.ndarray, bool]:
    """Returns (rows, has_objectness) where rows is (N, 4+nc) or (N, 5+nc)."""
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D detection output (after dropping the batch dim), got shape {raw.shape}")
    d0, d1 = raw.shape
    no_obj, with_obj = 4 + num_classes, 5 + num_classes
    if d0 == no_obj and d0 != d1:
        return raw.T, False
    if d0 == with_obj and d0 != d1:
        return raw.T, True
    if d1 == no_obj:
        return raw, False
    if d1 == with_obj:
        return raw, True
    raise ValueError(
        f"Detection output shape {raw.shape} doesn't match {num_classes} classes "
        f"(expected a {no_obj} or {with_obj} row/column count somewhere)."
    )


def predict_detection(
    model: ModelInfo,
    image_path: str,
    num_classes: int,
    class_mapping: dict[int, str],
    confidence_threshold: float,
    iou_threshold: float = 0.45,
    input_size: int | None = None,
) -> list[Prediction]:
    image = read_image_bgr(image_path)
    if image is None or num_classes <= 0:
        return []

    if model.format == FORMAT_ULTRALYTICS:
        return _predict_ultralytics(model, image, class_mapping, confidence_threshold, iou_threshold)

    orig_h, orig_w = image.shape[:2]

    size = input_size or model.input_width or model.input_height or DEFAULT_SIZE
    padded, scale, pad_x, pad_y = letterbox(image, size)
    tensor = bgr_to_tensor(padded, normalize_imagenet=False)

    outputs = model.run(tensor)
    raw = np.asarray(outputs[0])
    if raw.ndim == 3:
        raw = raw[0]
    rows, has_objectness = _normalize_rows(raw, num_classes)
    if rows.size == 0:
        return []

    boxes_xywh: list[list[float]] = []
    scores: list[float] = []
    class_indices: list[int] = []
    for row in rows:
        cx, cy, w, h = (float(v) for v in row[:4])
        if has_objectness:
            objectness = float(row[4])
            class_scores = row[5:]
            class_index = int(np.argmax(class_scores))
            confidence = objectness * float(class_scores[class_index])
        else:
            class_scores = row[4:]
            class_index = int(np.argmax(class_scores))
            confidence = float(class_scores[class_index])
        if confidence < confidence_threshold:
            continue
        boxes_xywh.append([cx - w / 2, cy - h / 2, w, h])
        scores.append(confidence)
        class_indices.append(class_index)

    if not boxes_xywh:
        return []

    keep_raw = cv2.dnn.NMSBoxes(boxes_xywh, scores, confidence_threshold, iou_threshold)
    keep = np.array(keep_raw).reshape(-1).astype(int).tolist() if len(keep_raw) else []

    predictions: list[Prediction] = []
    for i in keep:
        class_id = class_mapping.get(class_indices[i])
        if class_id is None:
            continue
        x, y, w, h = boxes_xywh[i]
        x1, y1, x2, y2 = x, y, x + w, y + h
        # Undo the letterbox transform: subtract padding, divide by scale.
        x1, y1 = (x1 - pad_x) / scale, (y1 - pad_y) / scale
        x2, y2 = (x2 - pad_x) / scale, (y2 - pad_y) / scale
        x1, x2 = sorted((max(0.0, min(x1, orig_w)), max(0.0, min(x2, orig_w))))
        y1, y2 = sorted((max(0.0, min(y1, orig_h)), max(0.0, min(y2, orig_h))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        predictions.append(
            Prediction(
                class_id=class_id,
                confidence=scores[i],
                points=[(x1 / orig_w, y1 / orig_h), (x2 / orig_w, y2 / orig_h)],
            )
        )
    return predictions


def build_predictor(
    model: ModelInfo, num_classes: int, class_mapping: dict[int, str], confidence: float, **options
) -> Callable[[str], list[Prediction]]:
    iou_threshold = options.get("iou_threshold", 0.45)
    input_size = options.get("input_size")

    def predict(image_path: str) -> list[Prediction]:
        return predict_detection(model, image_path, num_classes, class_mapping, confidence, iou_threshold, input_size)

    return predict


def apply_predictions(controller: ProjectController, image_id: str, predictions: list[Prediction], **options) -> None:
    for prediction in predictions:
        controller.add_annotation(image_id, prediction.class_id, prediction.points)
