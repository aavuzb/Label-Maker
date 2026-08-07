"""Classification workspace: whole-image labeling.

Three columns — Classes (left) | Images (middle, largest: filter, assign,
grid) | Preview (right, zoomable). Assigning selected images to the active
class immediately copies or moves their files into that class's folder
inside the project directory; there is no separate export step and no
manual save (every change autosaves).
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget

from app.features.classification.export_dialog import ExportDialog
from app.features.project.manager import MODE_COPY, MODE_MOVE, ProjectController
from app.shared.autolabel import classification as autolabel_classification
from app.shared.autolabel.dialog import AutoLabelRunDialog, AutoLabelSettings, AutoLabelSettingsDialog, AutoLabelTask
from app.shared.labeling.class_picker import ClassPickerPopup
from app.shared.labeling.class_toggle_panel import ClassTogglePanel
from app.shared.labeling.filter_bar import FilterBar
from app.shared.labeling.grid_widget import ImageGridWidget
from app.shared.labeling.image_model import (
    MAX_THUMBNAIL_SIZE,
    MIN_THUMBNAIL_SIZE,
    THUMBNAIL_SIZE,
    ImageFilterProxyModel,
    ImageListModel,
)
from app.shared.labeling.image_removal import confirm_and_remove
from app.shared.labeling.import_controls import build_import_row
from app.shared.labeling.panel_widgets import build_zoom_row, card_frame, panel_header
from app.shared.labeling.progress_widget import ProgressWidget
from app.shared.labeling.viewer_widget import ImageViewerWidget
from app.shared.theme import ACCENT, BG_SURFACE, FS_SMALL, outline_button_style, primary_button_style

_RESERVED_SHORTCUT_KEYS = {"A", "D", "F"}
_GRID_ZOOM_STEP = 24

_AUTOLABEL_DESCRIPTION = (
    "Runs a model you've already trained across your images and assigns the highest-confidence class "
    "automatically — handy if you have a model from a similar dataset instead of labeling from scratch. "
    "Expects a standard image classifier: one image in, one score per class out."
)

_MODE_BUTTON_STYLE = f"""
    QPushButton {{ border-radius: 6px; padding: 7px 16px; font-weight: 600; font-size: {FS_SMALL}px; }}
    QPushButton:checked {{ background-color: #fef3c7; border: 1px solid #d97706; color: #92400e; }}
    QPushButton:!checked {{ background-color: {BG_SURFACE}; border: 1px solid {ACCENT}; color: {ACCENT}; }}
"""


class ClassificationWorkspace(QWidget):
    def __init__(self, controller: ProjectController) -> None:
        super().__init__()
        self._controller = controller
        self._class_shortcuts: list[QShortcut] = []
        self._grid_icon_size = THUMBNAIL_SIZE
        self._autolabel_settings: AutoLabelSettings | None = None
        self._current_image_id: str | None = None

        self._model = ImageListModel(controller)
        self._proxy = ImageFilterProxyModel()
        self._proxy.setSourceModel(self._model)

        self.filter_bar = FilterBar(controller)
        self.class_panel = ClassTogglePanel(controller)
        self.viewer = ImageViewerWidget()
        self.grid = ImageGridWidget(show_edit_action=True)
        self.grid.setModel(self._proxy)
        self.progress = ProgressWidget(controller)

        self.filter_bar.filter_changed.connect(self._proxy.set_filter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 10)
        top_row = build_import_row(controller, self)
        autolabel_settings_button = QPushButton("Auto-Label Settings…")
        autolabel_settings_button.setStyleSheet(outline_button_style(ACCENT))
        autolabel_settings_button.setToolTip("Choose a model and options for automatic labeling")
        autolabel_settings_button.clicked.connect(self._open_autolabel_settings)
        top_row.addWidget(autolabel_settings_button)
        self.start_autolabel_button = QPushButton("Start Auto-Labeling")
        self.start_autolabel_button.setStyleSheet(outline_button_style(ACCENT))
        self.start_autolabel_button.setEnabled(False)
        self.start_autolabel_button.clicked.connect(self._start_autolabeling)
        top_row.addWidget(self.start_autolabel_button)
        self._update_start_autolabel_button()
        export_button = QPushButton("Export Dataset")
        export_button.setStyleSheet(outline_button_style(ACCENT))
        export_button.setToolTip("Export labeled images as an ImageFolder or CSV-manifest dataset")
        export_button.clicked.connect(self._open_export_dialog)
        top_row.addWidget(export_button)
        layout.addLayout(top_row)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self._build_classes_panel())
        main_splitter.addWidget(self._build_images_panel())
        main_splitter.addWidget(self._build_preview_panel())
        main_splitter.setSizes([220, 720, 320])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)
        main_splitter.setHandleWidth(10)

        layout.addWidget(main_splitter, stretch=1)
        layout.addWidget(self.progress)

        self.grid.assign_requested.connect(self._assign_selected)
        self.grid.remove_requested.connect(self._remove_selected)
        self.grid.edit_requested.connect(self._open_class_picker)
        self.grid.zoom_requested.connect(lambda direction: self._zoom_grid(direction * _GRID_ZOOM_STEP))
        self.grid.current_changed.connect(self._on_current_changed)
        controller.image_updated.connect(self._on_image_updated)

        QShortcut(QKeySequence("F"), self, activated=self.viewer.fit_to_screen)

        controller.project_opened.connect(self._rebuild_class_shortcuts)
        controller.classes_changed.connect(self._rebuild_class_shortcuts)
        self._rebuild_class_shortcuts()

    # -- left: classes -----------------------------------------------------

    def _build_classes_panel(self) -> QWidget:
        card = card_frame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 10, 14)
        layout.setSpacing(6)
        layout.addWidget(panel_header("CLASSES", "One class per image", accent=ACCENT))
        layout.addWidget(self.class_panel, stretch=1)
        return card

    # -- middle: images (largest panel) -------------------------------------

    def _build_images_panel(self) -> QWidget:
        assign_button = QPushButton("Assign (Enter)")
        assign_button.setStyleSheet(primary_button_style(ACCENT))
        assign_button.setToolTip("Move or copy the selected image(s) into the active class's folder")
        assign_button.clicked.connect(self._assign_selected)

        self.mode_button = QPushButton()
        self.mode_button.setCheckable(True)
        self.mode_button.setChecked(False)  # unchecked = Copy (default, non-destructive)
        self.mode_button.setStyleSheet(_MODE_BUTTON_STYLE)
        self.mode_button.setToolTip("Toggle whether Assign copies or moves the original file(s)")
        self.mode_button.toggled.connect(self._update_mode_button_text)
        self._update_mode_button_text()

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(assign_button)
        action_row.addWidget(self.mode_button)
        action_row.addStretch()
        action_row.addLayout(build_zoom_row(lambda: self._zoom_grid(-_GRID_ZOOM_STEP), lambda: self._zoom_grid(_GRID_ZOOM_STEP)))

        card = card_frame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 14)
        layout.setSpacing(8)
        layout.addWidget(panel_header("IMAGES", "Select, then Assign to the active class", accent=ACCENT))
        layout.addWidget(self.filter_bar)
        layout.addLayout(action_row)
        layout.addWidget(self.grid, stretch=1)
        return card

    def _zoom_grid(self, delta: int) -> None:
        new_size = max(MIN_THUMBNAIL_SIZE, min(MAX_THUMBNAIL_SIZE, self._grid_icon_size + delta))
        if new_size == self._grid_icon_size:
            return
        self._grid_icon_size = new_size
        self.grid.setIconSize(QSize(new_size, new_size))
        self._model.set_thumbnail_size(new_size)

    def _update_mode_button_text(self) -> None:
        if self.mode_button.isChecked():
            self.mode_button.setText("Mode: Move (originals removed)")
        else:
            self.mode_button.setText("Mode: Copy (originals kept)")

    def _current_mode(self) -> str:
        return MODE_MOVE if self.mode_button.isChecked() else MODE_COPY

    # -- right: preview -----------------------------------------------

    def _build_preview_panel(self) -> QWidget:
        card = card_frame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 14, 14)
        layout.setSpacing(6)
        layout.addWidget(panel_header("PREVIEW", accent=ACCENT))
        layout.addWidget(self.viewer, stretch=1)
        layout.addLayout(build_zoom_row(self.viewer.zoom_out, self.viewer.zoom_in, self.viewer.fit_to_screen))
        return card

    # -- assignment -----------------------------------------------

    def _on_current_changed(self, image_id: str) -> None:
        self._current_image_id = image_id or None
        project = self._controller.project
        entry = project.find_image(self._current_image_id) if project and self._current_image_id else None
        self.viewer.set_image_path(entry.path if entry else None)
        self._update_preview_class_badge(entry)

    def _on_image_updated(self, image_id: str) -> None:
        """Keeps the preview badge in sync when the currently-viewed image's
        class changes without re-selecting it — e.g. assigning it via the
        picker popup, or auto-labeling landing on it — without reloading the
        image itself (which would reset any zoom/pan the user has set up)."""
        if image_id != self._current_image_id:
            return
        project = self._controller.project
        entry = project.find_image(image_id) if project else None
        self._update_preview_class_badge(entry)

    def _update_preview_class_badge(self, entry) -> None:
        project = self._controller.project
        label = project.find_class(entry.class_id) if project and entry and entry.class_id else None
        self.viewer.set_class_label(label.name if label else None, label.color if label else None)

    def _assign_selected(self) -> None:
        project = self._controller.project
        if project is None or project.active_class_id is None:
            return
        image_ids = self.grid.selected_image_ids()
        if not image_ids:
            return
        self._assign_to_class(image_ids, project.active_class_id, "Assign")

    def _open_class_picker(self, global_pos) -> None:
        project = self._controller.project
        if project is None or not project.classes:
            return
        image_ids = self.grid.selected_image_ids()
        if not image_ids:
            return
        popup = ClassPickerPopup(project.classes, self)
        popup.class_picked.connect(lambda class_id, ids=image_ids: self._assign_to_class(ids, class_id, "Edit Class"))
        popup.popup_at(global_pos)

    def _assign_to_class(self, image_ids: list[str], class_id: str, action_label: str) -> None:
        failures = self._controller.assign_images(image_ids, class_id, self._current_mode())
        # Assigned images drop out of the (default) Unlabeled filter view on
        # their own via the proxy's dynamic filtering; nudge it explicitly
        # too so this doesn't silently depend on that Qt default staying on.
        self._proxy.invalidateFilter()
        if failures:
            QMessageBox.warning(
                self,
                action_label,
                f"{len(failures)} of {len(image_ids)} image(s) could not be assigned:\n"
                + "\n".join(f"- {message}" for _id, message in failures[:5]),
            )

    def _remove_selected(self) -> None:
        image_ids = self.grid.selected_image_ids()
        if confirm_and_remove(self, self._controller, image_ids):
            self._on_current_changed(self.grid.current_image_id() or "")

    def _open_export_dialog(self) -> None:
        ExportDialog(self._controller, self).exec()

    def _build_autolabel_task(self) -> AutoLabelTask:
        return AutoLabelTask(
            task_type="classification",
            accent=ACCENT,
            title="Classification",
            description=_AUTOLABEL_DESCRIPTION,
            default_input_size=autolabel_classification.DEFAULT_SIZE,
            build_predictor=autolabel_classification.build_predictor,
            apply_predictions=autolabel_classification.apply_predictions,
            needs_mode=True,
            needs_normalize_toggle=True,
        )

    def _open_autolabel_settings(self) -> None:
        dialog = AutoLabelSettingsDialog(self._controller, self._build_autolabel_task(), self._autolabel_settings, self)
        if dialog.exec():
            self._autolabel_settings = dialog.settings()
            self._update_start_autolabel_button()

    def _start_autolabeling(self) -> None:
        if self._autolabel_settings is None:
            return
        AutoLabelRunDialog(self._controller, self._build_autolabel_task(), self._autolabel_settings, self).exec()

    def _update_start_autolabel_button(self) -> None:
        settings = self._autolabel_settings
        self.start_autolabel_button.setEnabled(settings is not None)
        if settings is None:
            self.start_autolabel_button.setToolTip("Set up Auto-Label Settings first")
        else:
            self.start_autolabel_button.setToolTip(f"Run automatic labeling — {settings.summary()}")

    def _rebuild_class_shortcuts(self) -> None:
        for shortcut in self._class_shortcuts:
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._class_shortcuts.clear()

        project = self._controller.project
        if project is None:
            return
        for label in project.classes:
            key = label.shortcut.strip().upper()
            if not key or key in _RESERVED_SHORTCUT_KEYS:
                continue
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda class_id=label.id: self._controller.set_active_class(class_id))
            self._class_shortcuts.append(shortcut)
