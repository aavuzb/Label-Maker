"""Auto-labeling for Segmentation: image -> per-class polygon outlines.

Expected model contract: a semantic-segmentation model — a single image
input tensor (NCHW), and a single per-pixel class-score output of shape
(1, C, H, W) or (1, H, W, C) (either orientation is auto-detected). C must
equal either the number of project classes, or that plus one (channel 0
then assumed to be a "background" class with no corresponding project
class). This is a different contract than instance-segmentation networks
(e.g. Mask R-CNN, YOLO-seg) — the output here is one shared per-pixel map,
not one mask per detected instance.

Each connected region of a class's predicted area becomes its own polygon
via contour extraction, so one image can still end up with several
separate shapes of the same class (e.g. two distinct cats).

Raw Ultralytics YOLO-seg checkpoints (ModelInfo.format == FORMAT_ULTRALYTICS,
see model.py) don't go through any of the above — they predict per-instance
masks, not a shared per-pixel map, so `_predict_ultralytics` routes those
through Ultralytics' own `.predict()` instead, converting each instance's
mask straight to a polygon via `Results.masks.xyn` (already normalized to
[0, 1] — no contour extraction needed here, Ultralytics did it already).
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from app.features.project.manager import ProjectController
from app.shared.autolabel.model import FORMAT_ULTRALYTICS, ModelInfo
from app.shared.autolabel.preprocess import read_image_bgr, to_input_tensor
from app.shared.autolabel.prediction import Prediction

DEFAULT_SIZE = 256
MIN_REGION_AREA_FRACTION = 0.001  # ignore specks smaller than 0.1% of the mask area
SIMPLIFY_EPSILON_FRACTION = 0.01  # cv2.approxPolyDP epsilon, as a fraction of the contour's perimeter


def _simplify_normalized_polygon(polygon: np.ndarray) -> np.ndarray:
    """Ultralytics' Results.masks.xyn is the raw mask contour, essentially
    unsimplified (typically hundreds of points per instance) — reduced here
    with the same cv2.approxPolyDP epsilon-as-perimeter-fraction the
    semantic-segmentation path below uses, so an Ultralytics-derived polygon
    isn't wildly denser than a hand-drawn or semantic-model-derived one in
    the same project. Epsilon is scale-invariant (a fraction of the
    contour's own perimeter), so this works the same in normalized [0, 1]
    coordinates as it does in pixel space."""
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    epsilon = SIMPLIFY_EPSILON_FRACTION * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)


def _predict_ultralytics(
    model: ModelInfo, image: np.ndarray, class_mapping: dict[int, str], confidence_threshold: float
) -> list[Prediction]:
    yolo = model.backend.yolo
    results = yolo.predict(image, conf=confidence_threshold, device=model.backend.device_str, verbose=False)
    result = results[0]
    if result.masks is None or result.boxes is None:
        return []

    predictions: list[Prediction] = []
    for polygon, confidence, class_index in zip(
        result.masks.xyn, result.boxes.conf.tolist(), result.boxes.cls.tolist()
    ):
        class_id = class_mapping.get(int(class_index))
        if class_id is None or len(polygon) < 3:
            continue
        simplified = _simplify_normalized_polygon(polygon)
        if len(simplified) < 3:
            continue
        points = [(float(x), float(y)) for x, y in simplified]
        predictions.append(Prediction(class_id=class_id, confidence=float(confidence), points=points))
    return predictions


def _channel_first_probs(raw: np.ndarray) -> np.ndarray:
    """Normalizes a (C,H,W)-or-(H,W,C) raw score map to (C,H,W) softmax probabilities."""
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3D segmentation output (after dropping the batch dim), got shape {raw.shape}")
    if raw.shape[0] > raw.shape[-1]:
        raw = np.transpose(raw, (2, 0, 1))
    shifted = raw - raw.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=0, keepdims=True)


def predict_segmentation(
    model: ModelInfo,
    image_path: str,
    num_classes: int,
    class_mapping: dict[int, str],
    confidence_threshold: float,
    normalize_imagenet: bool = True,
    input_size: int | None = None,
) -> list[Prediction]:
    image = read_image_bgr(image_path)
    if image is None or num_classes <= 0:
        return []

    if model.format == FORMAT_ULTRALYTICS:
        return _predict_ultralytics(model, image, class_mapping, confidence_threshold)

    width = input_size or model.input_width or DEFAULT_SIZE
    height = input_size or model.input_height or DEFAULT_SIZE
    tensor = to_input_tensor(image, width, height, normalize_imagenet)

    outputs = model.run(tensor)
    raw = np.asarray(outputs[0])
    if raw.ndim == 4:
        raw = raw[0]
    probs = _channel_first_probs(raw)
    channels, mask_h, mask_w = probs.shape

    if channels == num_classes + 1:
        class_offset = 1  # channel 0 is background, channels 1..nc are real classes
    elif channels == num_classes:
        class_offset = 0
    else:
        raise ValueError(
            f"Segmentation output has {channels} channels, which matches neither "
            f"{num_classes} classes nor {num_classes + 1} (with a background channel)."
        )

    predicted = np.argmax(probs, axis=0)
    confidence_map = np.max(probs, axis=0)
    min_area = MIN_REGION_AREA_FRACTION * mask_h * mask_w

    predictions: list[Prediction] = []
    for model_index in range(class_offset, channels):
        output_index = model_index - class_offset
        class_id = class_mapping.get(output_index)
        if class_id is None:
            continue
        region = (predicted == model_index).astype(np.uint8)
        if not region.any():
            continue
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            region_mask = np.zeros_like(region)
            cv2.drawContours(region_mask, [contour], -1, 1, thickness=-1)
            region_confidence = float(confidence_map[region_mask.astype(bool)].mean())
            if region_confidence < confidence_threshold:
                continue
            epsilon = SIMPLIFY_EPSILON_FRACTION * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(simplified) < 3:
                continue
            points = [(float(x) / mask_w, float(y) / mask_h) for x, y in simplified]
            predictions.append(Prediction(class_id=class_id, confidence=region_confidence, points=points))
    return predictions


def build_predictor(
    model: ModelInfo, num_classes: int, class_mapping: dict[int, str], confidence: float, **options
) -> Callable[[str], list[Prediction]]:
    normalize_imagenet = options.get("normalize_imagenet", True)
    input_size = options.get("input_size")

    def predict(image_path: str) -> list[Prediction]:
        return predict_segmentation(model, image_path, num_classes, class_mapping, confidence, normalize_imagenet, input_size)

    return predict


def apply_predictions(controller: ProjectController, image_id: str, predictions: list[Prediction], **options) -> None:
    for prediction in predictions:
        controller.add_annotation(image_id, prediction.class_id, prediction.points)
