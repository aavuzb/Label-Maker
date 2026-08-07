"""Auto-labeling for Classification: whole-image -> one class.

Expected model contract: a single input image tensor (NCHW), a single
output of shape (1, num_classes) — raw logits or already-softmaxed
probabilities, either is fine (auto-detected).

Raw Ultralytics YOLO-classify checkpoints (ModelInfo.format ==
FORMAT_ULTRALYTICS, see model.py) skip this contract entirely and go
through `_predict_ultralytics`, which calls the checkpoint's own
`.predict()` instead of the tensor-in/tensor-out path above.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from app.features.project.manager import MODE_COPY, ProjectController
from app.shared.autolabel.model import FORMAT_ULTRALYTICS, ModelInfo
from app.shared.autolabel.preprocess import read_image_bgr, to_input_tensor
from app.shared.autolabel.prediction import Prediction

DEFAULT_SIZE = 224


def _predict_ultralytics(
    model: ModelInfo, image: np.ndarray, class_mapping: dict[int, str], confidence_threshold: float
) -> list[Prediction]:
    probs = model.backend.yolo.predict(image, device=model.backend.device_str, verbose=False)[0].probs
    if probs is None:
        return []
    data = probs.data.detach().cpu().numpy().reshape(-1)
    best_index = int(np.argmax(data))
    confidence = float(data[best_index])
    class_id = class_mapping.get(best_index)
    if class_id is None or confidence < confidence_threshold:
        return []
    return [Prediction(class_id=class_id, confidence=confidence, points=None)]


def _to_probabilities(logits: np.ndarray) -> np.ndarray:
    row = logits.astype(np.float64)
    # Already probabilities (softmax applied by the model itself)? Detect
    # rather than assume — re-softmaxing real probabilities would distort
    # their values and throw off the confidence threshold.
    looks_like_probs = bool(np.all(row >= -1e-6) and np.all(row <= 1 + 1e-6) and abs(row.sum() - 1.0) < 1e-3)
    if looks_like_probs:
        return np.clip(row, 0.0, 1.0)
    shifted = row - row.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def predict_classification(
    model: ModelInfo,
    image_path: str,
    class_mapping: dict[int, str],
    confidence_threshold: float,
    normalize_imagenet: bool = True,
    input_size: int | None = None,
) -> list[Prediction]:
    image = read_image_bgr(image_path)
    if image is None:
        return []

    if model.format == FORMAT_ULTRALYTICS:
        return _predict_ultralytics(model, image, class_mapping, confidence_threshold)

    width = input_size or model.input_width or DEFAULT_SIZE
    height = input_size or model.input_height or DEFAULT_SIZE
    tensor = to_input_tensor(image, width, height, normalize_imagenet)

    outputs = model.run(tensor)
    logits = np.asarray(outputs[0]).reshape(-1)
    probs = _to_probabilities(logits)

    best_index = int(np.argmax(probs))
    confidence = float(probs[best_index])
    class_id = class_mapping.get(best_index)
    if class_id is None or confidence < confidence_threshold:
        return []
    return [Prediction(class_id=class_id, confidence=confidence, points=None)]


def build_predictor(
    model: ModelInfo, num_classes: int, class_mapping: dict[int, str], confidence: float, **options
) -> Callable[[str], list[Prediction]]:
    normalize_imagenet = options.get("normalize_imagenet", True)
    input_size = options.get("input_size")

    def predict(image_path: str) -> list[Prediction]:
        return predict_classification(model, image_path, class_mapping, confidence, normalize_imagenet, input_size)

    return predict


def apply_predictions(controller: ProjectController, image_id: str, predictions: list[Prediction], **options) -> None:
    """Whole-image task: at most one prediction, applied exactly like a manual Assign."""
    if not predictions:
        return
    best = max(predictions, key=lambda p: p.confidence)
    mode = options.get("mode", MODE_COPY)
    controller.assign_images([image_id], best.class_id, mode)
