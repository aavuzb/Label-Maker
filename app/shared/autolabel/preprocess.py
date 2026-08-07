"""Image loading and model-input tensor preparation.

Two resize strategies are used on purpose:
  - `to_input_tensor` (plain stretch resize) for classification and semantic
    segmentation — accuracy doesn't depend on preserving aspect ratio, and
    for segmentation the output mask is read back in the *same* stretched
    space and converted straight to fractional [0,1] polygon coordinates,
    which undoes the stretch exactly.
  - `letterbox` (aspect-preserving resize + padding) for detection — YOLO
    models are near-universally trained and exported expecting letterboxed
    input, and box regression accuracy suffers if that convention isn't
    matched at inference time.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LETTERBOX_PAD_VALUE = 114  # the shade YOLO's own preprocessing conventionally pads with


def read_image_bgr(path: str) -> np.ndarray | None:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def bgr_to_tensor(image_bgr: np.ndarray, normalize_imagenet: bool = False) -> np.ndarray:
    """BGR uint8 HxWx3 -> NCHW float32 tensor, RGB, scaled to [0, 1] (and
    optionally ImageNet mean/std normalized) with a leading batch dim."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if normalize_imagenet:
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def to_input_tensor(image_bgr: np.ndarray, width: int, height: int, normalize_imagenet: bool = False) -> np.ndarray:
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    return bgr_to_tensor(resized, normalize_imagenet)


def letterbox(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resizes with aspect ratio preserved, padding to a `size` x `size`
    square. Returns (padded_image, scale, pad_x, pad_y) so box coordinates
    predicted in the padded space can be mapped back to the original image:
        original = (padded - pad) / scale
    """
    h, w = image_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), LETTERBOX_PAD_VALUE, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y
