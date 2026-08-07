"""Loads a trained model for auto-labeling from whichever format is on
hand, and exposes a single uniform `run()` interface regardless of the
underlying framework — so classification.py/detection.py/segmentation.py
never need to know which backend actually produced a prediction.

Supported formats (auto-detected from the file's extension):

  - .onnx                        ONNX, via onnxruntime. The recommended
                                  format: self-contained (architecture +
                                  weights in one file) and needs nothing
                                  beyond what LabelMaker already installs.
  - .weights  (+ a .cfg file)    Darknet — the classic YOLOv3/v4 pretrained
                                  weight format — via OpenCV's DNN module.
  - .caffemodel (+ a .prototxt)  Caffe, via OpenCV's DNN module.
  - .pb                          TensorFlow *frozen graph*, via OpenCV's
                                  DNN module. Not a SavedModel directory —
                                  those bundle variables separately and
                                  aren't a single file.
  - .pt / .pth                   PyTorch. Three different things can produce
                                  this extension, and all three are
                                  supported (tried in this order):
                                    - A TorchScript export (torch.jit.script/
                                      trace) — self-contained the same way
                                      the other formats are.
                                    - A raw Ultralytics YOLO checkpoint (the
                                      file `model.train()` or `yolo train`
                                      writes to runs/.../weights/best.pt) —
                                      loaded via the `ultralytics` package if
                                      it's installed. detect, classify, and
                                      segment checkpoints are all usable —
                                      each routes to the matching workspace's
                                      auto-labeler (detect->Detection,
                                      classify->Classification, segment
                                      ->Segmentation). A YOLO-seg checkpoint's
                                      per-instance masks are converted straight
                                      to polygons via Ultralytics' own
                                      mask-to-contour output (`Results.masks.
                                      xyn`), one polygon per detected
                                      instance — a different origin than the
                                      per-pixel semantic map a non-Ultralytics
                                      segmentation model produces, but the
                                      same Prediction(class_id, confidence,
                                      points) shape once decoded, so
                                      segmentation.py doesn't need to
                                      distinguish them. Other tasks (pose,
                                      obb, ...) still aren't a match for any
                                      auto-labeler here.
                                    - For Classification only: a plain
                                      torch.save() of a state_dict alongside
                                      a metadata field naming its own
                                      architecture (e.g. {"state_dict": ...,
                                      "model_name": "resnet18"} — the shape a
                                      lot of external training scripts save).
                                      A recognized torchvision architecture
                                      name is rebuilt via torchvision and the
                                      saved weights are loaded into it; the
                                      number of classes is inferred from the
                                      final layer's own shape. An
                                      unrecognized architecture, or a bare
                                      state_dict/nn.Module with no
                                      architecture metadata at all, still
                                      isn't loadable — no generic tool can
                                      rebuild a network's architecture from
                                      its weights alone; only the training
                                      code that defines that class can.
                                  All three routes require PyTorch installed
                                  separately (`pip install torch`); Ultralytics
                                  additionally needs `pip install ultralytics`,
                                  and the state_dict route additionally needs
                                  `pip install torchvision`. None are bundled
                                  by default since they're very large
                                  packages, so support for this extension is
                                  opt-in.

Darknet and Caffe split architecture and weights across two files, so
`load_model` takes an optional second `config_path`.

GPU is never used automatically — `load_model`'s `device` parameter
("cuda" or "cpu") is the caller's explicit choice, surfaced as the Device
dropdown in Auto-Label Settings. See gpu.py for hardware GPU detection
(independent of whether the installed onnxruntime/torch happen to be
CPU-only builds) and live busy/free-memory status.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.shared.autolabel.cpu_budget import cpu_thread_budget

FORMAT_ONNX = "onnx"
FORMAT_DARKNET = "darknet"
FORMAT_CAFFE = "caffe"
FORMAT_TENSORFLOW = "tensorflow"
FORMAT_TORCHSCRIPT = "torchscript"
FORMAT_ULTRALYTICS = "ultralytics"
FORMAT_TORCH_STATE_DICT = "torch-state-dict"

FORMAT_LABELS = {
    FORMAT_ONNX: "ONNX",
    FORMAT_DARKNET: "Darknet",
    FORMAT_CAFFE: "Caffe",
    FORMAT_TENSORFLOW: "TensorFlow (frozen graph)",
    FORMAT_TORCHSCRIPT: "PyTorch (TorchScript)",
    FORMAT_ULTRALYTICS: "Ultralytics YOLO (.pt checkpoint)",
    FORMAT_TORCH_STATE_DICT: "PyTorch (reconstructed from state_dict)",
}

# One entry per model-file extension we recognize. .pt/.pth is provisional
# here — load_model() actually tries TorchScript first and falls back to a
# raw Ultralytics checkpoint, so the ModelInfo it returns may end up tagged
# FORMAT_ULTRALYTICS instead. This entry only drives the config-file-needed
# check in the dialog (neither .pt route needs one), not the final label.
_EXTENSION_FORMATS = {
    ".onnx": FORMAT_ONNX,
    ".weights": FORMAT_DARKNET,
    ".caffemodel": FORMAT_CAFFE,
    ".pb": FORMAT_TENSORFLOW,
    ".pt": FORMAT_TORCHSCRIPT,
    ".pth": FORMAT_TORCHSCRIPT,
}

# Maps an Ultralytics checkpoint's own `.task` to the AutoLabelTask.task_type
# strings used by dialog.py/classification.py/detection.py/segmentation.py.
# Tasks with no entry (pose, obb, ...) aren't loadable here — no auto-labeler
# in this app has a matching output contract for them.
_ULTRALYTICS_TASK_TO_APP_TASK = {
    "detect": "detection",
    "classify": "classification",
    "segment": "segmentation",
}
_APP_TASK_LABELS = {
    "detection": "Detection",
    "classification": "Classification",
    "segmentation": "Segmentation",
}

CONFIG_REQUIRED_FORMATS = {FORMAT_DARKNET, FORMAT_CAFFE}
CONFIG_FILE_FILTERS = {
    FORMAT_DARKNET: "Darknet Config (*.cfg);;All Files (*)",
    FORMAT_CAFFE: "Caffe Config (*.prototxt);;All Files (*)",
}

MODEL_FILE_FILTER = (
    "All Supported Models (*.onnx *.weights *.caffemodel *.pb *.pt *.pth);;"
    "ONNX (*.onnx);;"
    "Darknet (*.weights);;"
    "Caffe (*.caffemodel);;"
    "TensorFlow Frozen Graph (*.pb);;"
    "PyTorch TorchScript (*.pt *.pth);;"
    "All Files (*)"
)


def _opencv_has_cuda() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except (AttributeError, cv2.error):
        return False  # this OpenCV build has no CUDA support at all


def _configure_opencv_net(net, has_cuda: bool):
    """Best-effort: point an OpenCV DNN net at CUDA if this OpenCV build
    actually has CUDA support compiled in (the `opencv-python-headless`
    wheel LabelMaker depends on normally doesn't — it's CPU-only — so this
    is mainly for anyone who's swapped in a CUDA-enabled OpenCV build
    themselves). Falls straight through to the CPU default otherwise."""
    if has_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def _configure_torch_cpu_threads(device: str) -> None:
    import torch

    # PyTorch's own default thread count is already not naive — it usually
    # tracks physical (not logical/hyperthreaded) cores, or an OMP/MKL env
    # var, so it can already be well under os.cpu_count(). Only ever shrink
    # it, never raise it — this must not undo a default that's already more
    # conservative than the plain "leave one core free" heuristic.
    if device != "cuda" or not torch.cuda.is_available():
        torch.set_num_threads(min(torch.get_num_threads(), cpu_thread_budget()))


def _torch_device(device: str):
    """`device` is the caller's explicit "cuda" or "cpu" choice (from the
    Auto-Label Settings device selector) — falls back to CPU if "cuda" was
    requested but this torch build/machine can't actually do it, rather
    than raising, since the UI already steers the user away from picking
    GPU when it isn't usable."""
    import torch

    return torch.device("cuda") if device == "cuda" and torch.cuda.is_available() else torch.device("cpu")


class _Backend(Protocol):
    def run(self, tensor: np.ndarray) -> list[np.ndarray]: ...


@dataclass
class ModelInfo:
    backend: _Backend
    format: str
    input_width: int | None
    input_height: int | None
    # Class names the model file itself declares, in output-index order —
    # populated when the format has a standard place for them (Ultralytics
    # checkpoints/exports, or a state_dict checkpoint that saved its own
    # class list). None when the format doesn't carry this information.
    class_names: list[str] | None = None
    # Set only for FORMAT_TORCH_STATE_DICT — the torchvision architecture
    # name it was reconstructed as, shown to confirm the guess was right.
    architecture: str | None = None
    # Whether this model actually ended up on a GPU backend/device/provider
    # (not just "is a GPU present somewhere on this machine") — every loader
    # below already prefers GPU automatically when the installed
    # library/build actually supports it; this just records what happened,
    # so the UI can warn before a slow, freeze-prone CPU run instead of
    # silently starting one.
    uses_gpu: bool = False

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.backend.run(tensor)

    def describe(self) -> str:
        label = FORMAT_LABELS.get(self.format, self.format)
        if self.architecture:
            label = f"{label} ({self.architecture})"
        if self.input_width and self.input_height:
            base = f"{label} model — input {self.input_width} × {self.input_height}"
        else:
            base = f"{label} model — input size isn't declared in the file; set it manually below"
        if self.class_names:
            base += f" — {len(self.class_names)} class name(s) found in the model"
        base += " — running on GPU" if self.uses_gpu else " — running on CPU (slower; may briefly strain your machine)"
        return base


class _OnnxBackend:
    def __init__(self, session, input_name: str) -> None:
        self._session = session
        self._input_name = input_name

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return list(self._session.run(None, {self._input_name: tensor}))


class _OpenCVDnnBackend:
    """Shared by Darknet, Caffe, and TensorFlow-frozen-graph — OpenCV's DNN
    module loads and runs all three through the same cv2.dnn.Net API."""

    def __init__(self, net) -> None:
        self._net = net

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        self._net.setInput(tensor)
        names = self._net.getUnconnectedOutLayersNames()
        outputs = self._net.forward(names)
        if isinstance(outputs, np.ndarray):
            return [outputs]
        return list(outputs)


class _TorchScriptBackend:
    def __init__(self, module, device: str) -> None:
        self._device = _torch_device(device)
        self._module = module.to(self._device)

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        import torch

        with torch.no_grad():
            output = self._module(torch.from_numpy(tensor).to(self._device))
        if isinstance(output, (tuple, list)):
            return [o.detach().cpu().numpy() for o in output]
        return [output.detach().cpu().numpy()]


class _UltralyticsBackend:
    """Wraps an `ultralytics.YOLO` instance. classification.py/detection.py/
    segmentation.py all reach past `.run()` and call `.yolo.predict(...)`
    directly for this format — Ultralytics does its own letterboxing and
    output decoding (which differs across YOLO versions: v8/v11 emit raw
    per-class scores, v10/v26-style heads already return post-NMS boxes;
    segment heads additionally return per-instance masks), so re-deriving
    that from a raw tensor here would mean re-implementing per-version
    decoding logic that Ultralytics already gets right. `.run()` is kept
    only so this still satisfies the `_Backend` protocol."""

    def __init__(self, yolo, device_str: str) -> None:
        self.yolo = yolo
        # Already resolved to a concrete "cuda"/"cpu" (not the raw request)
        # — callers pass this straight to `.predict(device=...)` so the
        # explicit Settings choice is honored there too, not just here.
        self.device_str = device_str

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        import torch

        with torch.no_grad():
            output = self.yolo.model(torch.from_numpy(tensor))
        while isinstance(output, (tuple, list)):
            output = output[0]
        return [output.detach().cpu().numpy()]


def detect_format(path: Path) -> str:
    fmt = _EXTENSION_FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"Unrecognized model file “{path.name}”. Supported: .onnx, .weights (+ .cfg), "
            ".caffemodel (+ .prototxt), .pb (TensorFlow frozen graph), .pt/.pth (PyTorch TorchScript "
            "or a raw Ultralytics YOLO checkpoint)."
        )
    return fmt


def load_model(
    path: Path, config_path: Path | None = None, expected_task: str | None = None, device: str = "cuda"
) -> ModelInfo:
    """Raises ValueError (or whatever the underlying framework raises, which
    is usually specific enough to show the user directly) on failure.

    `expected_task` is the AutoLabelTask.task_type ("classification",
    "detection", or "segmentation") of the dialog doing the loading. It's
    only used to sanity-check Ultralytics checkpoints, which carry their own
    task and can otherwise "load" successfully in the wrong window and then
    silently produce zero predictions.

    `device` is the user's explicit "cuda" or "cpu" choice from the
    Auto-Label Settings device selector — every loader below only attempts
    GPU when this is "cuda" (and falls back to CPU on its own if the
    installed package/hardware can't actually do it), never automatically,
    since a "cpu" choice must be honored even on a machine that does have a
    working GPU.
    """
    # Image preprocessing (preprocess.py's resize/color-convert calls) goes
    # through OpenCV regardless of which backend does inference, and OpenCV
    # defaults to using every logical core for it — capped globally here,
    # once, rather than only for the OpenCV-DNN-backed formats below.
    cv2.setNumThreads(cpu_thread_budget())

    fmt = detect_format(path)
    if fmt == FORMAT_ONNX:
        return _load_onnx(path, device)
    if fmt == FORMAT_DARKNET:
        return _load_darknet(path, config_path, device)
    if fmt == FORMAT_CAFFE:
        return _load_caffe(path, config_path, device)
    if fmt == FORMAT_TENSORFLOW:
        return _load_tensorflow(path, device)
    return _load_pytorch(path, expected_task, device)


def _load_onnx(path: Path, device: str) -> ModelInfo:
    import onnxruntime as ort

    # Prefer GPU (needs the separate `onnxruntime-gpu` package — the plain
    # `onnxruntime` LabelMaker depends on by default is CPU-only and simply
    # won't offer CUDAExecutionProvider here) when the user asked for it,
    # and fall back to CPU otherwise — never automatically, so a "cpu"
    # choice from the Settings device selector is always honored even when
    # a working GPU is available. Either way, cap the CPU-side thread pool:
    # onnxruntime defaults to using every logical core, which is exactly
    # what makes a long CPU run peg the whole machine instead of just this
    # app.
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = cpu_thread_budget()
    session_options.inter_op_num_threads = 1

    available_providers = ort.get_available_providers()
    want_gpu = device == "cuda" and "CUDAExecutionProvider" in available_providers
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if want_gpu else ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(path), sess_options=session_options, providers=providers)
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("This model declares no inputs — is it a valid ONNX model?")
    input_tensor = inputs[0]
    shape = list(input_tensor.shape)
    return ModelInfo(
        backend=_OnnxBackend(session, input_tensor.name),
        format=FORMAT_ONNX,
        input_width=_static_dim(shape, -1),
        input_height=_static_dim(shape, -2),
        class_names=_read_onnx_class_names(session),
        # get_providers() (not the `providers` list requested above) is the
        # ground truth — onnxruntime silently falls back to CPU if the CUDA
        # provider fails to actually initialize (missing CUDA/cuDNN
        # libraries, mismatched versions, etc.) even when it was requested.
        uses_gpu="CUDAExecutionProvider" in session.get_providers(),
    )


def _read_onnx_class_names(session) -> list[str] | None:
    """Ultralytics' ONNX export (`yolo export format=onnx`) embeds the
    model's class names as a stringified {index: name} dict in the model's
    custom metadata — other exporters don't have a standard place for this,
    so most ONNX models simply won't carry any."""
    try:
        metadata = session.get_modelmeta().custom_metadata_map
        raw = metadata.get("names")
        if not raw:
            return None
        parsed = ast.literal_eval(raw)
    except Exception:  # noqa: BLE001 — best-effort; absent/malformed metadata isn't an error
        return None
    if isinstance(parsed, dict) and parsed:
        try:
            return [str(parsed[key]) for key in sorted(parsed, key=lambda k: int(k))]
        except (KeyError, ValueError, TypeError):
            return [str(v) for v in parsed.values()]
    if isinstance(parsed, (list, tuple)) and parsed:
        return [str(v) for v in parsed]
    return None


def _load_darknet(path: Path, config_path: Path | None, device: str) -> ModelInfo:
    if config_path is None:
        raise ValueError("A Darknet .weights file also needs its matching .cfg config file.")
    use_cuda = device == "cuda" and _opencv_has_cuda()
    net = _configure_opencv_net(cv2.dnn.readNetFromDarknet(str(config_path), str(path)), use_cuda)
    width, height = _parse_darknet_size(config_path)
    return ModelInfo(
        backend=_OpenCVDnnBackend(net), format=FORMAT_DARKNET, input_width=width, input_height=height, uses_gpu=use_cuda
    )


def _load_caffe(path: Path, config_path: Path | None, device: str) -> ModelInfo:
    if config_path is None:
        raise ValueError("A Caffe .caffemodel file also needs its matching .prototxt config file.")
    use_cuda = device == "cuda" and _opencv_has_cuda()
    net = _configure_opencv_net(cv2.dnn.readNetFromCaffe(str(config_path), str(path)), use_cuda)
    return ModelInfo(
        backend=_OpenCVDnnBackend(net), format=FORMAT_CAFFE, input_width=None, input_height=None, uses_gpu=use_cuda
    )


def _load_tensorflow(path: Path, device: str) -> ModelInfo:
    use_cuda = device == "cuda" and _opencv_has_cuda()
    net = _configure_opencv_net(cv2.dnn.readNetFromTensorflow(str(path)), use_cuda)
    return ModelInfo(
        backend=_OpenCVDnnBackend(net), format=FORMAT_TENSORFLOW, input_width=None, input_height=None, uses_gpu=use_cuda
    )


def _load_pytorch(path: Path, expected_task: str | None, device: str) -> ModelInfo:
    """.pt/.pth is ambiguous on its own — try the self-contained TorchScript
    route first, and if that's not what this file is, fall back to treating
    it as a raw Ultralytics YOLO checkpoint (the far more common thing for a
    freshly-trained .pt to actually be)."""
    try:
        import torch
    except ImportError as exc:
        raise ValueError(
            "Loading a .pt/.pth model needs PyTorch installed separately (pip install torch) — "
            "LabelMaker doesn't bundle it by default since it's a very large package. Exporting your "
            "model to ONNX instead (torch.onnx.export, or `yolo export format=onnx` for a YOLO model) "
            "works without installing anything extra."
        ) from exc

    # GPU is only used when explicitly requested (see _torch_device/
    # _TorchScriptBackend, and the device passed to Ultralytics below) —
    # this only matters for whichever part stays on CPU, so it doesn't
    # default to pegging every core for the whole run.
    _configure_torch_cpu_threads(device)

    try:
        module = torch.jit.load(str(path), map_location="cpu")
    except Exception as torchscript_exc:  # noqa: BLE001 — torch raises several different exception types here
        return _load_ultralytics(path, torchscript_exc, expected_task, device)

    module.eval()
    resolved_device = _torch_device(device)
    return ModelInfo(
        backend=_TorchScriptBackend(module, device),
        format=FORMAT_TORCHSCRIPT,
        input_width=None,
        input_height=None,
        uses_gpu=resolved_device.type == "cuda",
    )


def _load_ultralytics(path: Path, torchscript_exc: Exception, expected_task: str | None, device: str) -> ModelInfo:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        if expected_task == "classification":
            classifier = _try_load_torchvision_classifier(path, device)
            if classifier is not None:
                return classifier
        raise ValueError(
            "This isn't a TorchScript export, so it was tried as a raw Ultralytics YOLO checkpoint "
            "instead (the usual thing a training run's .pt file actually is) — but that needs the "
            "`ultralytics` package, which isn't installed (pip install ultralytics). Alternatively, "
            "export it yourself first — `yolo export model=... format=onnx` — and load the .onnx file."
        ) from exc

    try:
        yolo = YOLO(str(path))
    except Exception as exc:  # noqa: BLE001 — ultralytics/torch raise several different exception types here
        if expected_task == "classification":
            classifier = _try_load_torchvision_classifier(path, device)
            if classifier is not None:
                return classifier
        state_dict_hint = _describe_if_bare_state_dict(path)
        if state_dict_hint is not None:
            raise ValueError(
                f"This .pt file holds {state_dict_hint}, not a self-contained model — neither "
                "TorchScript nor Ultralytics can load it, because no generic tool can rebuild a "
                "network's architecture from its weights alone; only the training code that defines "
                "that class can. Export it to ONNX from that training code instead "
                "(torch.onnx.export(model, example_input, \"model.onnx\")) and load the .onnx file here."
            ) from exc
        raise ValueError(
            f"Couldn't load this .pt file as TorchScript ({torchscript_exc}) or as an Ultralytics "
            f"YOLO checkpoint ({exc}). Export it to ONNX instead — `yolo export model=... "
            f"format=onnx` for a YOLO model — and load that file."
        ) from exc

    app_task = _ULTRALYTICS_TASK_TO_APP_TASK.get(yolo.task)
    if app_task is None:
        raise ValueError(
            f"This is a YOLO “{yolo.task}” checkpoint. LabelMaker's auto-labeler can load detect, "
            "classify, and segment checkpoints directly; other tasks (pose, obb, ...) aren't a match "
            "for any auto-labeler here. Export a compatible ONNX model instead, or use a "
            "detect/classify/segment checkpoint."
        )
    if expected_task is not None and app_task != expected_task:
        raise ValueError(
            f"This is a “{yolo.task}”-task Ultralytics YOLO checkpoint, which belongs in the "
            f"{_APP_TASK_LABELS.get(app_task, app_task)} workspace's Auto-Label window, not "
            f"{_APP_TASK_LABELS.get(expected_task, expected_task)}."
        )

    yolo.model.eval()
    size = _ultralytics_input_size(yolo)
    resolved_device = _torch_device(device)
    device_str = resolved_device.type

    return ModelInfo(
        backend=_UltralyticsBackend(yolo, device_str),
        format=FORMAT_ULTRALYTICS,
        input_width=size,
        input_height=size,
        class_names=_read_ultralytics_class_names(yolo),
        # `.predict()` (what classification.py/detection.py actually call —
        # see _UltralyticsBackend's docstring) is passed device_str
        # explicitly there, so it honors this same choice instead of
        # auto-selecting CUDA on its own.
        uses_gpu=device_str == "cuda",
    )


def _read_ultralytics_class_names(yolo) -> list[str] | None:
    names = getattr(yolo, "names", None)
    if isinstance(names, dict) and names:
        try:
            return [str(names[key]) for key in sorted(names, key=lambda k: int(k))]
        except (KeyError, ValueError, TypeError):
            return [str(v) for v in names.values()]
    if isinstance(names, (list, tuple)) and names:
        return [str(v) for v in names]
    return None


def _ultralytics_input_size(yolo) -> int | None:
    imgsz = getattr(yolo.model, "args", None)
    imgsz = imgsz.get("imgsz") if isinstance(imgsz, dict) else None
    if isinstance(imgsz, (list, tuple)):
        imgsz = imgsz[0] if imgsz else None
    return int(imgsz) if isinstance(imgsz, (int, float)) else None


def _is_tensor_dict(value) -> bool:
    import torch

    return isinstance(value, dict) and bool(value) and all(isinstance(v, torch.Tensor) for v in value.values())


def _describe_if_bare_state_dict(path: Path) -> str | None:
    """If `path` turns out to be a plain torch.save() of a state_dict — with
    or without a metadata dict wrapped around it, e.g. {"state_dict": ...,
    "model_name": ...} — this gives a specific, actionable description of
    what was found instead of a model. Used only to improve the error
    message once TorchScript, Ultralytics, and (for Classification) the
    torchvision reconstruction below have already failed to load the file;
    returns None if torch.load() itself fails, or the file isn't shaped like
    a state_dict, since then there's nothing more specific to say than
    "couldn't load it" anyway."""
    try:
        import torch

        data = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 — best-effort diagnosis, not the real load attempt
        return None

    if isinstance(data, dict) and _is_tensor_dict(data.get("state_dict")):
        name = data.get("model_name") or data.get("arch") or data.get("architecture")
        return f"a state_dict (its own metadata names the architecture “{name}”)" if name else "a state_dict"
    if _is_tensor_dict(data):
        return "a bare state_dict"
    return None


# -- Classification-only: reconstructing a state_dict + architecture-name checkpoint --

_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "net", "model")
_ARCH_METADATA_KEYS = ("model_name", "arch", "architecture", "backbone")
_CLASS_NAME_KEYS = ("class_names", "classes", "labels", "categories")
_ARCH_NAME_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_arch_name(name: str) -> str:
    return _ARCH_NAME_NORMALIZE_RE.sub("", name.lower())


def _resolve_torchvision_architecture(name: str) -> str | None:
    import torchvision.models as tv_models

    normalized = _normalize_arch_name(name)
    for candidate in tv_models.list_models(module=tv_models):
        if _normalize_arch_name(candidate) == normalized:
            return candidate
    return None


def _infer_num_classes(state_dict: dict) -> int | None:
    """The final classification layer is almost always the last entry
    written into a model's state_dict (PyTorch preserves definition order),
    so the last 1D bias (or last 2D weight matrix's row count) is a
    reliable, architecture-agnostic way to recover how many classes a
    checkpoint was trained for — without needing to know that architecture's
    specific layer names."""
    for key, tensor in reversed(list(state_dict.items())):
        if key.endswith("bias") and tensor.dim() == 1:
            return int(tensor.shape[0])
        if key.endswith("weight") and tensor.dim() == 2:
            return int(tensor.shape[0])
    return None


def _read_checkpoint_class_names(checkpoint: dict, num_classes: int) -> list[str] | None:
    for key in _CLASS_NAME_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, (list, tuple)) and len(value) == num_classes:
            return [str(v) for v in value]
    class_to_idx = checkpoint.get("class_to_idx")
    if isinstance(class_to_idx, dict) and len(class_to_idx) == num_classes:
        try:
            return [name for name, _idx in sorted(class_to_idx.items(), key=lambda kv: int(kv[1]))]
        except (TypeError, ValueError):
            return None
    return None


def _try_load_torchvision_classifier(path: Path, device: str) -> ModelInfo | None:
    """Reconstructs a plain torch.save()'d classifier checkpoint — a
    state_dict plus a metadata field naming its architecture, e.g.
    {"state_dict": ..., "model_name": "resnet18"} — by rebuilding that named
    architecture via torchvision and loading the saved weights into it.
    Only called for Classification (see `_load_ultralytics`'s callers):
    torchvision's classifier registry isn't a match for Detection/
    Segmentation's model contracts here.

    Returns None if the file doesn't even look like a metadata-tagged
    state_dict, so the caller falls through to its own generic error
    message. Once it does look like one, failures raise ValueError with a
    specific, actionable reason instead of returning None — a checkpoint
    that clearly declares itself a "resnet18" state_dict but can't be
    reconstructed deserves a better error than "couldn't load it".
    """
    import torch

    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 — not a valid torch.save() at all; let the caller's own error stand
        return None
    if not isinstance(checkpoint, dict):
        return None

    arch_name = next((checkpoint[key] for key in _ARCH_METADATA_KEYS if checkpoint.get(key)), None)
    state_dict = next(
        (checkpoint[key] for key in _STATE_DICT_KEYS if _is_tensor_dict(checkpoint.get(key))), None
    )
    if state_dict is None and _is_tensor_dict(checkpoint):
        state_dict = checkpoint
    if not arch_name or state_dict is None:
        return None
    arch_name = str(arch_name)

    try:
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            f"This .pt file holds a state_dict for a “{arch_name}” model — reconstructing it needs the "
            "torchvision package (pip install torchvision), which isn't installed."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — installed but broken, usually a torch/torchvision version mismatch
        raise ValueError(
            f"This .pt file holds a state_dict for a “{arch_name}” model — reconstructing it needs "
            f"torchvision, which is installed but failed to load ({exc}). This almost always means the "
            "installed torch and torchvision versions weren't built for each other — reinstall a matching "
            "pair, e.g. `pip install --force-reinstall torch torchvision` (or install both together from "
            "https://pytorch.org/get-started/locally/ for a guaranteed-compatible pair)."
        ) from exc

    canonical = _resolve_torchvision_architecture(arch_name)
    if canonical is None:
        raise ValueError(
            f"This .pt file holds a state_dict naming its architecture as “{arch_name}”, which isn't one "
            "of torchvision's built-in classifiers — LabelMaker can only reconstruct well-known "
            "architectures (ResNet, VGG, DenseNet, MobileNet, EfficientNet, …) this way. Export it to "
            "ONNX from the original training code instead (torch.onnx.export(model, example_input, "
            "\"model.onnx\")) and load the .onnx file here."
        )

    # DataParallel/DistributedDataParallel saves prefix every key with
    # "module." — strip it so the keys line up with a freshly-built model.
    clean_state_dict = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    num_classes = _infer_num_classes(clean_state_dict)
    if num_classes is None:
        raise ValueError(
            f"Couldn't work out how many classes this “{arch_name}” checkpoint was trained for from its "
            "weight shapes."
        )

    from torchvision.models import get_model

    model = get_model(canonical, weights=None, num_classes=num_classes)
    try:
        model.load_state_dict(clean_state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"This .pt file names its architecture as “{arch_name}”, but its saved weights don't line up "
            f"with torchvision's {canonical} layer-for-layer ({exc}). It may be a modified variant of the "
            "architecture, which can't be reconstructed generically — export it to ONNX instead."
        ) from exc
    model.eval()
    resolved_device = _torch_device(device)

    return ModelInfo(
        backend=_TorchScriptBackend(model, device),
        format=FORMAT_TORCH_STATE_DICT,
        input_width=None,
        input_height=None,
        architecture=canonical,
        class_names=_read_checkpoint_class_names(checkpoint, num_classes),
        uses_gpu=resolved_device.type == "cuda",
    )


def _static_dim(shape: list, index: int) -> int | None:
    if not shape or index >= len(shape) or -index > len(shape):
        return None
    value = shape[index]
    return value if isinstance(value, int) and value > 0 else None


_DARKNET_DIM_RE = re.compile(r"^(width|height)\s*=\s*(\d+)", re.IGNORECASE)


def _parse_darknet_size(config_path: Path) -> tuple[int | None, int | None]:
    """Darknet .cfg files declare their expected input size as plain-text
    `width=...` / `height=...` lines in the [net] section."""
    width = height = None
    try:
        for line in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = _DARKNET_DIM_RE.match(line.strip())
            if not match:
                continue
            if match.group(1).lower() == "width":
                width = int(match.group(2))
            else:
                height = int(match.group(2))
            if width and height:
                break
    except OSError:
        pass
    return width, height
