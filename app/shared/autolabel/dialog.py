"""Shared Auto-Label UI — one implementation parametrized per task via an
AutoLabelTask spec, since the model picker, class mapping, scope, and run
machinery are identical across Classification/Detection/Segmentation and
only a handful of options actually differ.

Split into two dialogs rather than one:
  - AutoLabelSettingsDialog — choose a model, class mapping, and options;
    "Save" just packages them into an AutoLabelSettings and hands it back,
    without touching the project or running anything.
  - AutoLabelRunDialog — takes an already-saved AutoLabelSettings and runs
    it immediately, showing progress. Re-running later (a second "Start
    Auto-Labeling" click) doesn't require reconfiguring or re-browsing for
    the model file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.features.project.manager import MODE_COPY, MODE_MOVE, ProjectController
from app.shared.autolabel import gpu as gpu_module
from app.shared.autolabel.model import (
    CONFIG_FILE_FILTERS,
    CONFIG_REQUIRED_FORMATS,
    MODEL_FILE_FILTER,
    ModelInfo,
    detect_format,
    load_model,
)
from app.shared.autolabel.prediction import Prediction, resolve_class_mapping, unlabeled_image_ids
from app.shared.autolabel.worker import AutoLabelWorker
from app.shared.labeling.panel_widgets import build_callout
from app.shared.theme import FS_CAPTION, primary_button_style

_SUCCESS_COLOR = "#16a34a"
_ERROR_COLOR = "#dc2626"

CLASS_MODE_MERGED = "merged"  # "Use this project's + the model's classes"
CLASS_MODE_MODEL_ONLY = "model_only"  # "Use the model's classes"


def _wrap(inner_layout) -> QWidget:
    """Wraps a layout in a real QWidget. Word-wrapped QLabels nested inside
    a layout that isn't itself owned by a widget can get an unstable
    heightForWidth calculation and end up overlapping whatever comes after
    them — wrapping in a QWidget fixes it (bit the landing page once already)."""
    widget = QWidget()
    widget.setLayout(inner_layout)
    return widget


@dataclass
class AutoLabelTask:
    task_type: str
    accent: str
    title: str
    description: str
    default_input_size: int
    build_predictor: Callable[..., Callable[[str], list[Prediction]]]
    apply_predictions: Callable[..., None]
    needs_iou: bool = False
    needs_mode: bool = False
    needs_normalize_toggle: bool = False
    duplicate_warning: str = ""  # shown when "All images" scope is picked, if non-empty


@dataclass
class AutoLabelSettings:
    """Everything AutoLabelRunDialog needs to run — captured once by
    AutoLabelSettingsDialog's Save, then reusable across as many "Start
    Auto-Labeling" clicks as the workspace stays open for. `model` keeps the
    already-loaded backend alive so running (or reopening Settings to review
    them) never needs to re-read the model file from disk."""

    model: ModelInfo
    model_path: str
    config_path: str | None
    class_mode: str  # CLASS_MODE_MERGED or CLASS_MODE_MODEL_ONLY
    confidence: float
    scope: str  # "unlabeled" or "all"
    device: str  # "cuda" or "cpu" — the user's explicit choice, not auto-detected
    options: dict = field(default_factory=dict)  # input_size / iou_threshold / mode / normalize_imagenet

    def summary(self) -> str:
        scope_label = "all images" if self.scope == "all" else "unlabeled images"
        device_label = "GPU" if self.device == "cuda" else "CPU"
        return f"{Path(self.model_path).name} — confidence {self.confidence:.2f} — {scope_label} — {device_label}"


class _ModelPickerMixin:
    """Model browsing/loading + class-mapping widgets, shared by
    AutoLabelSettingsDialog. Not used by AutoLabelRunDialog, which only ever
    reads an already-loaded ModelInfo off a saved AutoLabelSettings."""

    _task: AutoLabelTask
    _model: ModelInfo | None
    _config_widgets: list[QWidget]
    _needs_input_size: bool

    def _build_model_section(self) -> QWidget:
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("Choose a trained model file…")
        browse_model_button = QPushButton("Browse…")
        browse_model_button.clicked.connect(self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path_edit, stretch=1)
        model_row.addWidget(browse_model_button)

        formats_hint = QLabel(
            "Supports ONNX (.onnx), Darknet (.weights + .cfg), Caffe (.caffemodel + .prototxt), "
            "TensorFlow frozen graphs (.pb), PyTorch TorchScript (.pt/.pth), raw Ultralytics YOLO "
            "checkpoints (.pt, needs the ultralytics package), and — for Classification only — a "
            "PyTorch state_dict checkpoint that names its own architecture (needs the torchvision package)."
        )
        formats_hint.setWordWrap(True)
        self._style_status(formats_hint, "muted")

        self.device_combo = QComboBox()
        self.device_combo.addItem("GPU (CUDA)", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

        self.gpu_status_label = QLabel("")
        self.gpu_status_label.setWordWrap(True)
        self._style_status(self.gpu_status_label, "muted")
        initial_gpu_status = gpu_module.detect_gpu()
        # Default to whichever the machine can actually do — GPU stays
        # selectable either way (installing the right package is still
        # something the user can go do), it just isn't the default pick on
        # a machine with no GPU hardware at all.
        self.device_combo.blockSignals(True)
        self.device_combo.setCurrentIndex(self.device_combo.findData("cuda" if initial_gpu_status.hardware_present else "cpu"))
        self.device_combo.blockSignals(False)
        self._update_gpu_status_label(initial_gpu_status)

        self.config_row_label = QLabel("Config file")
        self.config_path_edit = QLineEdit()
        self.config_path_edit.setReadOnly(True)
        self.browse_config_button = QPushButton("Browse…")
        self.browse_config_button.clicked.connect(self._browse_config)
        config_row = QHBoxLayout()
        config_row.addWidget(self.config_path_edit, stretch=1)
        config_row.addWidget(self.browse_config_button)
        self._config_row_widget = _wrap(config_row)

        self.model_status_label = QLabel("No model loaded yet.")
        self.model_status_label.setWordWrap(True)
        self._style_status(self.model_status_label, "muted")

        form = QFormLayout()
        form.addRow("Model", model_row)
        form.addRow("", formats_hint)
        form.addRow("Device", self.device_combo)
        form.addRow("", self.gpu_status_label)
        form.addRow(self.config_row_label, self._config_row_widget)
        form.addRow("", self.model_status_label)
        self.config_row_label.setVisible(False)
        self._config_row_widget.setVisible(False)
        self._config_widgets += [browse_model_button, self.browse_config_button, self.device_combo]
        return _wrap(form)

    def _update_gpu_status_label(self, status: gpu_module.GpuStatus) -> None:
        self.gpu_status_label.setText(status.describe())
        self._style_status(self.gpu_status_label, "success" if status.usable_now else "muted")

    def _selected_device(self) -> str:
        return self.device_combo.currentData() or "cpu"

    def _on_device_changed(self) -> None:
        self._update_gpu_status_label(gpu_module.detect_gpu())
        # A model already loaded was placed on whatever device was
        # selected *at that time* — reload it now so switching the
        # dropdown takes effect immediately instead of silently only
        # applying the next time a model file is browsed.
        if self._model is not None:
            model_path = self.model_path_edit.text().strip()
            if model_path:
                config_path = self.config_path_edit.text().strip()
                self._try_load_model(Path(model_path), Path(config_path) if config_path else None)

    def _build_class_mapping_section(self) -> QWidget:
        self.use_merged_classes_radio = QRadioButton("Use this project's + the model's classes")
        self.use_merged_classes_radio.setChecked(True)
        self.use_model_classes_radio = QRadioButton("Use the model's classes")
        self.use_model_classes_radio.setEnabled(False)

        self.class_names_status_label = QLabel("Choose a model above to see which classes it declares.")
        self.class_names_status_label.setWordWrap(True)
        self._style_status(self.class_names_status_label, "muted")

        note = QLabel(
            "Class names are read directly from the model file when it declares them (an Ultralytics "
            "YOLO checkpoint/export, or a Classification state_dict checkpoint that saves its own class "
            "list) — there's no file to browse for. A name that matches an existing project class "
            "(case-insensitive) is reused as-is; any other name is created automatically. If the model "
            "doesn't declare names, only the first option works, falling back to pairing the model's "
            "output order with this project's classes in order."
        )
        note.setWordWrap(True)
        self._style_status(note, "muted")

        # A QFormLayout, not a plain QVBoxLayout — a hand-rolled QVBoxLayout
        # mixing addWidget()/addLayout() calls with a word-wrapped QLabel
        # got an unstable heightForWidth calculation and overlapped the row
        # after it; QFormLayout (used everywhere else in this dialog)
        # doesn't have that problem.
        form = QFormLayout()
        form.addRow(QLabel("Which classes does the model predict?"))
        form.addRow(self.use_merged_classes_radio)
        form.addRow(self.use_model_classes_radio)
        form.addRow("", self.class_names_status_label)
        form.addRow("", note)
        self._config_widgets += [self.use_merged_classes_radio, self.use_model_classes_radio]
        return _wrap(form)

    def _build_options_section(self) -> QWidget:
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(0.9)
        confidence_hint = QLabel("Only keep predictions the model is at least this confident about. Higher = fewer, more reliable labels.")
        confidence_hint.setWordWrap(True)
        self._style_status(confidence_hint, "muted")

        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Only unlabeled images (safe default)", "unlabeled")
        self.scope_combo.addItem("All images", "all")

        form = QFormLayout()
        form.addRow("Confidence threshold", self.confidence_spin)
        form.addRow("", confidence_hint)
        form.addRow("Apply to", self.scope_combo)
        self._config_widgets += [self.confidence_spin, self.scope_combo]

        if self._task.duplicate_warning:
            self.scope_warning_label = QLabel(self._task.duplicate_warning)
            self.scope_warning_label.setWordWrap(True)
            self._style_status(self.scope_warning_label, "muted")
            self.scope_warning_label.setVisible(False)
            self.scope_combo.currentIndexChanged.connect(
                lambda: self.scope_warning_label.setVisible(self.scope_combo.currentData() == "all")
            )
            form.addRow("", self.scope_warning_label)

        if self._task.needs_iou:
            self.iou_spin = QDoubleSpinBox()
            self.iou_spin.setRange(0.05, 0.95)
            self.iou_spin.setSingleStep(0.05)
            self.iou_spin.setDecimals(2)
            self.iou_spin.setValue(0.45)
            form.addRow("Overlap (IoU) threshold", self.iou_spin)
            self._config_widgets.append(self.iou_spin)

        if self._task.needs_mode:
            self.mode_combo = QComboBox()
            self.mode_combo.addItem("Copy (originals kept)", "copy")
            self.mode_combo.addItem("Move (originals moved)", "move")
            form.addRow("File mode", self.mode_combo)
            self._config_widgets.append(self.mode_combo)

        if self._task.needs_normalize_toggle:
            self.normalize_checkbox = QCheckBox("Normalize like ImageNet (recommended for most transfer-learned models)")
            self.normalize_checkbox.setChecked(True)
            form.addRow("", self.normalize_checkbox)
            self._config_widgets.append(self.normalize_checkbox)

        self.input_size_spin = QSpinBox()
        # Floor of 1, not some larger "sane minimum": QSpinBox's default
        # correction mode reverts an out-of-range typed value back to
        # whatever it held *before* editing (not clamped to the nearest
        # bound) once focus leaves the field — so a bound here that's higher
        # than a legitimately small model input (e.g. 28 for MNIST-style
        # classifiers) silently discards the user's edit instead of erroring.
        self.input_size_spin.setRange(1, 4096)
        self.input_size_spin.setValue(self._task.default_input_size)
        self.input_size_spin.setSuffix(" px")
        self.input_size_spin.setButtonSymbols(QAbstractSpinBox.PlusMinus)
        self.input_size_row_label = QLabel("Input size (model doesn't declare a fixed size)")
        form.addRow(self.input_size_row_label, self.input_size_spin)
        self.input_size_row_label.setVisible(False)
        self.input_size_spin.setVisible(False)
        self._config_widgets.append(self.input_size_spin)

        return _wrap(form)

    @staticmethod
    def _style_status(label: QLabel, kind: str) -> None:
        color = {"muted": None, "success": _SUCCESS_COLOR, "error": _ERROR_COLOR}[kind]
        from app.shared.theme import TEXT_MUTED

        label.setStyleSheet(f"color: {color or TEXT_MUTED}; font-size: {FS_CAPTION}px; border: none;")

    # -- model loading -----------------------------------------------

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose Model", "", MODEL_FILE_FILTER)
        if not path:
            return
        self.model_path_edit.setText(path)
        self.config_path_edit.setText("")
        self._model = None

        try:
            fmt = detect_format(Path(path))
        except ValueError as exc:
            self.model_status_label.setText(str(exc))
            self._style_status(self.model_status_label, "error")
            self.config_row_label.setVisible(False)
            self._config_row_widget.setVisible(False)
            return

        needs_config = fmt in CONFIG_REQUIRED_FORMATS
        self.config_row_label.setVisible(needs_config)
        self._config_row_widget.setVisible(needs_config)
        if needs_config:
            self.model_status_label.setText("Now choose the matching config file below to finish loading this model.")
            self._style_status(self.model_status_label, "muted")
            return

        self._try_load_model(Path(path), None)

    def _browse_config(self) -> None:
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            return
        fmt = detect_format(Path(model_path))
        file_filter = CONFIG_FILE_FILTERS.get(fmt, "All Files (*)")
        path, _ = QFileDialog.getOpenFileName(self, "Choose Config File", "", file_filter)
        if not path:
            return
        self.config_path_edit.setText(path)
        self._try_load_model(Path(model_path), Path(path))

    def _try_load_model(self, model_path: Path, config_path: Path | None) -> None:
        try:
            self._model = load_model(
                model_path, config_path, expected_task=self._task.task_type, device=self._selected_device()
            )
        except Exception as exc:  # noqa: BLE001 — surface the framework's own message, it's usually specific
            self._model = None
            self.model_status_label.setText(f"Couldn't load this model: {exc}")
            self._style_status(self.model_status_label, "error")
            self._needs_input_size = False
            self.input_size_row_label.setVisible(False)
            self.input_size_spin.setVisible(False)
            self._update_class_names_status()
            return

        self.model_status_label.setText(f"Model loaded — {self._model.describe()}")
        self._style_status(self.model_status_label, "success")

        # Tracked as its own flag rather than read back via
        # input_size_spin.isVisible(): Qt widgets only report themselves as
        # visible once their top-level window has actually been shown, so
        # isVisible() is unreliable here in _prefill(), which runs from
        # __init__ before the dialog is shown — checking it there silently
        # dropped a previously-saved input_size back to the default.
        self._needs_input_size = self._model.input_width is None or self._model.input_height is None
        self.input_size_row_label.setVisible(self._needs_input_size)
        self.input_size_spin.setVisible(self._needs_input_size)
        self._update_class_names_status()

    def _update_class_names_status(self) -> None:
        class_names = self._model.class_names if self._model else None
        self.use_model_classes_radio.setEnabled(bool(class_names))
        if class_names:
            preview = ", ".join(class_names[:5]) + ("…" if len(class_names) > 5 else "")
            self.class_names_status_label.setText(f"This model declares {len(class_names)} class name(s): {preview}")
            self._style_status(self.class_names_status_label, "success")
            return

        if self.use_model_classes_radio.isChecked():
            self.use_merged_classes_radio.setChecked(True)
        if self._model is None:
            self.class_names_status_label.setText("Choose a model above to see which classes it declares.")
        else:
            self.class_names_status_label.setText(
                "This model doesn't declare its own class names, so “Use the model's classes” isn't "
                "available — only “Use this project's + the model's classes” works, falling back to "
                "pairing the model's output order with this project's classes in order."
            )
        self._style_status(self.class_names_status_label, "muted")


class AutoLabelSettingsDialog(_ModelPickerMixin, QDialog):
    """Choose a model, class mapping, and run options; Save just packages
    them up. Doesn't touch the project or run anything — that's
    AutoLabelRunDialog's job, once "Start Auto-Labeling" is clicked."""

    def __init__(
        self,
        controller: ProjectController,
        task: AutoLabelTask,
        initial: AutoLabelSettings | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._task = task
        self._model: ModelInfo | None = None
        self._settings: AutoLabelSettings | None = None
        self._needs_input_size = False

        self.setWindowTitle(f"Auto-Label Settings — {task.title}")
        self.setMinimumWidth(660)
        self.resize(700, 600)

        self._config_widgets: list[QWidget] = []

        # The configurable part is scrollable rather than assumed to always
        # fit on screen — with every optional row visible (IoU, mode,
        # normalize toggle, dynamic input size) this can get tall, and a
        # dialog that just clips or overlaps its own rows on a small
        # display is worse than one with a scrollbar.
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)
        scroll_layout.addWidget(self._build_model_section())
        scroll_layout.addWidget(self._build_class_mapping_section())
        scroll_layout.addWidget(self._build_options_section())
        scroll_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addWidget(build_callout(task.description, task.accent))
        layout.addWidget(scroll_area, stretch=1)
        layout.addWidget(self._build_buttons())

        if initial is not None:
            self._prefill(initial)

    def _build_buttons(self) -> QWidget:
        self.save_button = QPushButton("Save Settings")
        self.save_button.setStyleSheet(primary_button_style(self._task.accent))
        self.save_button.clicked.connect(self._save)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(cancel_button)
        row.addWidget(self.save_button)
        wrapper = QWidget()
        wrapper.setLayout(row)
        return wrapper

    # -- prefill (reopening Settings after a previous Save) -----------------

    def _prefill(self, initial: AutoLabelSettings) -> None:
        self.model_path_edit.setText(initial.model_path)
        if initial.config_path:
            self.config_path_edit.setText(initial.config_path)
            self.config_row_label.setVisible(True)
            self._config_row_widget.setVisible(True)

        # Blocked: the model below is already loaded on this device (kept
        # alive on the settings object) — letting this fire _on_device_changed
        # would trigger a pointless reload of the exact same thing.
        self.device_combo.blockSignals(True)
        device_index = self.device_combo.findData(initial.device)
        if device_index >= 0:
            self.device_combo.setCurrentIndex(device_index)
        self.device_combo.blockSignals(False)
        self._update_gpu_status_label(gpu_module.detect_gpu())

        # The model's already loaded (kept alive on the settings object) —
        # no need to re-read it from disk just to review/edit other options.
        self._model = initial.model
        self.model_status_label.setText(f"Model loaded — {self._model.describe()}")
        self._style_status(self.model_status_label, "success")
        self._needs_input_size = self._model.input_width is None or self._model.input_height is None
        self.input_size_row_label.setVisible(self._needs_input_size)
        self.input_size_spin.setVisible(self._needs_input_size)
        self._update_class_names_status()

        if initial.class_mode == CLASS_MODE_MODEL_ONLY:
            self.use_model_classes_radio.setChecked(True)
        else:
            self.use_merged_classes_radio.setChecked(True)

        self.confidence_spin.setValue(initial.confidence)
        scope_index = self.scope_combo.findData(initial.scope)
        if scope_index >= 0:
            self.scope_combo.setCurrentIndex(scope_index)

        if self._task.needs_iou and "iou_threshold" in initial.options:
            self.iou_spin.setValue(initial.options["iou_threshold"])
        if self._task.needs_mode and "mode" in initial.options:
            mode_index = self.mode_combo.findData(initial.options["mode"])
            if mode_index >= 0:
                self.mode_combo.setCurrentIndex(mode_index)
        if self._task.needs_normalize_toggle and "normalize_imagenet" in initial.options:
            self.normalize_checkbox.setChecked(initial.options["normalize_imagenet"])
        if self._needs_input_size and "input_size" in initial.options:
            self.input_size_spin.setValue(initial.options["input_size"])

    # -- save -----------------------------------------------

    def _save(self) -> None:
        if self._model is None:
            QMessageBox.warning(
                self, "Auto-Label Settings", "Choose a valid model first (and its config file, if the format needs one)."
            )
            return

        class_names = self._model.class_names
        if self.use_model_classes_radio.isChecked() and not class_names:
            QMessageBox.warning(
                self,
                "Auto-Label Settings",
                "This model doesn't declare its own class names, so “Use the model's classes” isn't "
                "available. Switch to “Use this project's + the model's classes” instead.",
            )
            return

        # Create the model's declared classes in the project as soon as
        # Settings are saved — not only once "Start Auto-Labeling" is
        # clicked — so they show up immediately in the class list on this
        # page and (same shared project) the other two workspaces. Held off
        # until Save rather than done the moment the model file loads, so
        # just browsing/comparing a few candidate models before deciding
        # doesn't dump every one of their class lists into the project.
        # Matches by name (case-insensitive) and never duplicates.
        if class_names and self._controller.project is not None:
            self._controller.import_classes(class_names)

        options: dict = {}
        if self._needs_input_size:
            options["input_size"] = self.input_size_spin.value()
        if self._task.needs_iou:
            options["iou_threshold"] = self.iou_spin.value()
        if self._task.needs_mode:
            options["mode"] = MODE_MOVE if self.mode_combo.currentData() == "move" else MODE_COPY
        if self._task.needs_normalize_toggle:
            options["normalize_imagenet"] = self.normalize_checkbox.isChecked()

        self._settings = AutoLabelSettings(
            model=self._model,
            model_path=self.model_path_edit.text().strip(),
            config_path=self.config_path_edit.text().strip() or None,
            class_mode=CLASS_MODE_MODEL_ONLY if self.use_model_classes_radio.isChecked() else CLASS_MODE_MERGED,
            confidence=self.confidence_spin.value(),
            scope=self.scope_combo.currentData(),
            device=self._selected_device(),
            options=options,
        )
        self.accept()

    def settings(self) -> AutoLabelSettings | None:
        return self._settings


class AutoLabelRunDialog(QDialog):
    """Runs a previously-saved AutoLabelSettings and shows progress. Starts
    itself automatically as soon as it's shown — clicking "Start
    Auto-Labeling" is already the user's go-ahead, so there's no separate
    confirmation step inside this dialog."""

    def __init__(self, controller: ProjectController, task: AutoLabelTask, settings: AutoLabelSettings, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._task = task
        self._settings = settings
        self._worker: AutoLabelWorker | None = None
        self._labeled_count = 0
        self._failed_count = 0
        self._total = 0
        self._batch_active = False

        self.setWindowTitle(f"Auto-Label — {task.title}")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addWidget(build_callout(task.description, task.accent))
        layout.addWidget(self._build_progress_section())
        layout.addWidget(self._build_buttons())

        # Deferred rather than called directly from __init__: this lets the
        # dialog actually paint and become visible first, so the user sees
        # it appear before any work (including the class-mapping mutation
        # below) starts.
        QTimer.singleShot(0, self._start)

    def _build_progress_section(self) -> QWidget:
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self._style_status(self.status_label, "muted")

        column = QVBoxLayout()
        column.addWidget(self.progress_bar)
        column.addWidget(self.status_label)
        return _wrap(column)

    def _build_buttons(self) -> QWidget:
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel)

        self.close_button = QPushButton("Close")
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.close)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.cancel_button)
        row.addWidget(self.close_button)
        wrapper = QWidget()
        wrapper.setLayout(row)
        return wrapper

    @staticmethod
    def _style_status(label: QLabel, kind: str) -> None:
        color = {"muted": None, "success": _SUCCESS_COLOR, "error": _ERROR_COLOR}[kind]
        from app.shared.theme import TEXT_MUTED

        label.setStyleSheet(f"color: {color or TEXT_MUTED}; font-size: {FS_CAPTION}px; border: none;")

    # -- run -----------------------------------------------

    def _start(self) -> None:
        project = self._controller.project
        if project is None:
            self.reject()
            return

        if self._settings.device == "cpu":
            proceed = QMessageBox.warning(
                self,
                "Auto-Label — CPU Selected",
                "CPU is selected for this run (in Auto-Label Settings). Running on CPU is much slower and "
                "can make your computer or monitor unresponsive for a while until it finishes.\n\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if proceed != QMessageBox.Yes:
                self.reject()
                return
        else:
            # A live check, not the snapshot from whenever Settings was
            # last opened — utilization/free memory change constantly as
            # other processes use the GPU, right up until this run starts.
            gpu_status = gpu_module.detect_gpu()
            if not gpu_status.usable_now:
                QMessageBox.warning(
                    self,
                    "Auto-Label — GPU Not Available",
                    f"{gpu_status.unusable_reason()}\n\n"
                    "Switch to CPU in Auto-Label Settings to run anyway — but note CPU can make your "
                    "computer or monitor unresponsive for a while until it finishes.",
                )
                self.reject()
                return

        num_classes, class_mapping = resolve_class_mapping(self._controller, self._settings.model.class_names)
        if not class_mapping:
            QMessageBox.warning(
                self,
                "Auto-Label",
                "No classes to label with — add at least one class first, or check the model's declared classes.",
            )
            self.reject()
            return

        project = self._controller.project  # re-fetch: resolve_class_mapping may have added classes
        scope = self._settings.scope
        image_ids = unlabeled_image_ids(project) if scope == "unlabeled" else [e.id for e in project.images]
        if not image_ids:
            QMessageBox.information(self, "Auto-Label", "There are no images to label in this scope.")
            self.reject()
            return
        wanted = set(image_ids)
        path_by_id = {e.id: e.path for e in project.images if e.id in wanted}

        predict_fn = self._task.build_predictor(
            self._settings.model, num_classes, class_mapping, self._settings.confidence, **self._settings.options
        )

        self._labeled_count = 0
        self._failed_count = 0
        self._total = len(image_ids)
        self.progress_bar.setMaximum(self._total)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Running on {self._total} image(s)…")
        self._style_status(self.status_label, "muted")

        self._worker = AutoLabelWorker(image_ids, path_by_id, predict_fn)
        self._worker.progress.connect(self._on_progress)
        self._worker.image_labeled.connect(self._on_image_labeled)
        self._worker.image_failed.connect(self._on_image_failed)
        self._worker.finished_all.connect(self._on_finished)
        # Each apply_predictions() call below autosaves by default (see
        # ProjectController) — fine for a single manual edit, but a full
        # project.json rewrite per prediction across a whole run blocks the
        # UI thread long enough to freeze the screen. Batched here and
        # flushed to one write in _on_finished/_stop_and_wait instead.
        self._controller.begin_batch()
        self._batch_active = True
        self._set_running(True)
        # LowPriority is a hint to the OS scheduler to favor the UI thread
        # whenever the two are actually contending for CPU time — cheap
        # insurance on top of the thread caps in model.py, for a machine
        # that starts to strain regardless.
        self._worker.start(QThread.LowPriority)

    def _on_progress(self, done: int, _total: int) -> None:
        self.progress_bar.setValue(done)

    def _on_image_labeled(self, image_id: str, predictions: list[Prediction]) -> None:
        self._labeled_count += 1
        self._task.apply_predictions(self._controller, image_id, predictions, **self._settings.options)

    def _on_image_failed(self, _image_id: str, _message: str) -> None:
        self._failed_count += 1

    def _end_batch_once(self) -> None:
        """Idempotent: _on_finished (normal/cancelled completion) and
        _stop_and_wait (dialog closed while the worker is still running)
        can both reach here, and the batch must be flushed exactly once."""
        if self._batch_active:
            self._batch_active = False
            self._controller.end_batch()

    def _on_finished(self, _labeled: int, processed: int) -> None:
        self._end_batch_once()
        self._set_running(False)
        summary = f"Labeled {self._labeled_count} of {processed} image(s)."
        if self._failed_count:
            summary += f" {self._failed_count} image(s) couldn't be read."
        if processed < self._total:
            summary += " Stopped early."
        self.status_label.setText(summary)
        self._style_status(self.status_label, "success" if self._labeled_count else "muted")
        self._worker = None

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Stopping…")
            self._style_status(self.status_label, "muted")

    def _set_running(self, running: bool) -> None:
        self.cancel_button.setEnabled(running)
        self.close_button.setEnabled(not running)

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
        # Safety net alongside _on_finished: if the dialog is closed while
        # the worker is still running, its queued finished_all signal may
        # never be delivered (Qt drops events for a receiver that's already
        # being torn down) — without this, the batch begun in _start()
        # would stay open forever and silently block every future autosave.
        self._end_batch_once()
