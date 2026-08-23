"""Segmentation workspace: per-object polygon (mask) annotation.

Three columns — Classes + Objects-in-image (left) | Images (middle: browse
imported images) | Canvas (right: draw/edit/delete polygons on the
selected image, zoomable). One image can hold objects of many different
classes, so there's no "active class" driving what a new shape gets tagged
with — closing a shape immediately opens a search-as-you-type class picker
anchored on it instead, and right-clicking an existing shape offers the
same picker to change its class, or delete it. Images are never copied or
moved — only annotation data (saved into project.json) changes.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget

from app.features.project.manager import ProjectController
from app.features.segmentation.canvas import SegmentationCanvas
from app.features.segmentation.export_dialog import ExportDialog
from app.shared.autolabel import segmentation as autolabel_segmentation
from app.shared.autolabel.dialog import AutoLabelRunDialog, AutoLabelSettings, AutoLabelSettingsDialog, AutoLabelTask
from app.shared.labeling.class_toggle_panel import ClassTogglePanel
from app.shared.labeling.filter_bar import FilterBar
from app.shared.labeling.grid_widget import ImageGridWidget
from app.shared.labeling.image_model import MAX_THUMBNAIL_SIZE, MIN_THUMBNAIL_SIZE, THUMBNAIL_SIZE, ImageFilterProxyModel, ImageListModel
from app.shared.labeling.image_removal import confirm_and_remove
from app.shared.labeling.import_controls import build_import_row
from app.shared.labeling.objects_list import ObjectsListPanel
from app.shared.labeling.panel_widgets import build_callout, build_zoom_row, card_frame, panel_header, status_pill
from app.shared.labeling.progress_widget import ProgressWidget
from app.shared.theme import TASK_ACCENTS, outline_button_style

_ACCENT = TASK_ACCENTS["segmentation"]
_GRID_ZOOM_STEP = 24

# Segmentation is the least familiar workflow of the three, so this spells
# every step out — what a "polygon" means here, how to place points, and
# exactly how to finish — instead of assuming prior labeling experience.
_INSTRUCTIONS = (
    "New here? A segmentation shape is just a chain of points traced around an object's outline:\n"
    "1.  Click once per corner/curve, working your way around the object's edge — hold Shift for the first "
    "point if it lands on top of an existing shape, to start a new one there anyway.\n"
    "2.  Click back on point 1 (it turns green when you're close), press Enter, or double-click to close it.\n"
    "3.  Pick its class from the search box that appears — type to filter, or click straight from the list.\n"
    "4.  Select a shape to drag it or any of its points; double-click a point to delete it.\n"
    "5.  Right-click a shape to change its class or delete it · Escape cancels a shape you're still drawing."
)

_AUTOLABEL_DESCRIPTION = (
    "Runs a trained semantic segmentation model across your images and traces a shape for every region it "
    "finds above the confidence threshold. Expects a model that scores every pixel by class (e.g. a "
    "DeepLab/U-Net-style model) — not an instance-segmentation model like Mask R-CNN or YOLO-seg, which "
    "score individual detected objects instead."
)
_AUTOLABEL_DUPLICATE_WARNING = "Running on images that already have shapes adds more on top — it won't replace or deduplicate the existing ones."


class SegmentationWorkspace(QWidget):
    def __init__(self, controller: ProjectController) -> None:
        super().__init__()
        self._controller = controller
        self._current_image_id: str | None = None
        self._autolabel_settings: AutoLabelSettings | None = None
        self._grid_icon_size = THUMBNAIL_SIZE

        self._model = ImageListModel(controller)
        self._proxy = ImageFilterProxyModel()
        self._proxy.setSourceModel(self._model)

        # Unlike classification, annotating an image shouldn't make it
        # disappear from the list mid-edit, so default to "All" here.
        self.filter_bar = FilterBar(controller, default_mode=ImageFilterProxyModel.FILTER_ALL)
        self.class_panel = ClassTogglePanel(controller)
        self.objects_panel = ObjectsListPanel(controller)
        self.canvas = SegmentationCanvas(controller)
        self.grid = ImageGridWidget()
        self.grid.setModel(self._proxy)
        self.progress = ProgressWidget(controller)

        self.filter_bar.filter_changed.connect(self._proxy.set_filter)
        self.objects_panel.object_selected.connect(self.canvas.select_annotation)
        self.objects_panel.delete_requested.connect(self._on_delete_object)
        self.objects_panel.class_change_requested.connect(self._on_change_object_class)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 10)
        top_row = build_import_row(controller, self)
        autolabel_settings_button = QPushButton("Auto-Label Settings…")
        autolabel_settings_button.setStyleSheet(outline_button_style(_ACCENT))
        autolabel_settings_button.setToolTip("Choose a model and options for automatic labeling")
        autolabel_settings_button.clicked.connect(self._open_autolabel_settings)
        top_row.addWidget(autolabel_settings_button)
        self.start_autolabel_button = QPushButton("Start Auto-Labeling")
        self.start_autolabel_button.setStyleSheet(outline_button_style(_ACCENT))
        self.start_autolabel_button.setEnabled(False)
        self.start_autolabel_button.clicked.connect(self._start_autolabeling)
        top_row.addWidget(self.start_autolabel_button)
        self._update_start_autolabel_button()
        export_button = QPushButton("Export Dataset")
        export_button.setStyleSheet(outline_button_style(_ACCENT))
        export_button.setToolTip("Export labeled images as a YOLO-seg or COCO dataset")
        export_button.clicked.connect(self._open_export_dialog)
        top_row.addWidget(export_button)
        layout.addLayout(top_row)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self._build_left_column())
        main_splitter.addWidget(self._build_images_panel())
        main_splitter.addWidget(self._build_canvas_panel())
        main_splitter.setSizes([260, 380, 700])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setStretchFactor(2, 1)
        main_splitter.setHandleWidth(10)

        layout.addWidget(main_splitter, stretch=1)
        layout.addWidget(self.progress)

        self.grid.current_changed.connect(self._on_current_changed)
        self.grid.selection_changed.connect(self._on_selection_changed)
        self.grid.remove_requested.connect(self._remove_selected)
        self.grid.zoom_requested.connect(lambda direction: self._zoom_grid(direction * _GRID_ZOOM_STEP))

        QShortcut(QKeySequence("F"), self, activated=self.canvas.fit_to_screen)

        controller.classes_changed.connect(self._on_classes_changed)

    # -- left: classes + objects-in-image -----------------------------------

    def _build_left_column(self) -> QWidget:
        classes_card = card_frame()
        classes_layout = QVBoxLayout(classes_card)
        classes_layout.setContentsMargins(14, 12, 10, 12)
        classes_layout.setSpacing(6)
        classes_layout.addWidget(panel_header("CLASSES", "Add and manage your project's classes", accent=_ACCENT))
        classes_layout.addWidget(self.class_panel, stretch=1)

        objects_card = card_frame()
        objects_layout = QVBoxLayout(objects_card)
        objects_layout.setContentsMargins(14, 12, 10, 12)
        objects_layout.setSpacing(6)
        objects_layout.addWidget(panel_header("OBJECTS IN THIS IMAGE", "Each keeps its own class", accent=_ACCENT))
        objects_layout.addWidget(self.objects_panel, stretch=1)

        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(classes_card)
        left_splitter.addWidget(objects_card)
        left_splitter.setSizes([260, 280])
        left_splitter.setHandleWidth(10)
        return left_splitter

    # -- middle: images -----------------------------------------------

    def _build_images_panel(self) -> QWidget:
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        filter_row.addWidget(self.filter_bar, stretch=1)
        filter_row.addLayout(build_zoom_row(lambda: self._zoom_grid(-_GRID_ZOOM_STEP), lambda: self._zoom_grid(_GRID_ZOOM_STEP)))

        card = card_frame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 14)
        layout.setSpacing(8)
        layout.addWidget(panel_header("IMAGES", accent=_ACCENT))
        layout.addLayout(filter_row)
        layout.addWidget(self.grid, stretch=1)
        return card

    def _zoom_grid(self, delta: int) -> None:
        new_size = max(MIN_THUMBNAIL_SIZE, min(MAX_THUMBNAIL_SIZE, self._grid_icon_size + delta))
        if new_size == self._grid_icon_size:
            return
        self._grid_icon_size = new_size
        self.grid.setIconSize(QSize(new_size, new_size))
        self._model.set_thumbnail_size(new_size)

    # -- right: canvas -----------------------------------------------

    def _build_canvas_panel(self) -> QWidget:
        card = card_frame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)
        layout.addWidget(panel_header("CANVAS", accent=_ACCENT))
        layout.addWidget(build_callout(_INSTRUCTIONS, _ACCENT))
        layout.addLayout(self._build_export_neighbor_row())
        layout.addWidget(self.canvas, stretch=1)
        self.status_label = status_pill(_ACCENT)
        layout.addWidget(self.status_label)
        layout.addLayout(build_zoom_row(self.canvas.zoom_out, self.canvas.zoom_in, self.canvas.fit_to_screen))
        self.canvas.status_changed.connect(self.status_label.setText)
        self.canvas.refresh_status()
        return card

    def _build_export_neighbor_row(self) -> QHBoxLayout:
        self.export_previous_button = QPushButton("Import from Previous")
        self.export_previous_button.setStyleSheet(outline_button_style(_ACCENT))
        self.export_previous_button.setToolTip("Copy the previous image's shapes and classes onto this image")
        self.export_previous_button.clicked.connect(lambda: self._export_from_neighbor(-1))
        self.export_next_button = QPushButton("Import from Next")
        self.export_next_button.setStyleSheet(outline_button_style(_ACCENT))
        self.export_next_button.setToolTip("Copy the next image's shapes and classes onto this image")
        self.export_next_button.clicked.connect(lambda: self._export_from_neighbor(1))

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.export_previous_button)
        row.addWidget(self.export_next_button)
        row.addStretch()
        return row

    # -- wiring -----------------------------------------------

    def _on_current_changed(self, image_id: str) -> None:
        self._current_image_id = image_id or None
        self.canvas.load_image(self._current_image_id)
        self.objects_panel.set_image(self._current_image_id)

    def _on_selection_changed(self, image_ids: list[str]) -> None:
        if len(image_ids) == 1 and image_ids[0] != self._current_image_id:
            self._current_image_id = image_ids[0]
            self.canvas.load_image(self._current_image_id)
            self.objects_panel.set_image(self._current_image_id)

    def _remove_selected(self) -> None:
        image_ids = self.grid.selected_image_ids()
        if confirm_and_remove(self, self._controller, image_ids):
            self._on_current_changed(self.grid.current_image_id() or "")

    def _open_export_dialog(self) -> None:
        ExportDialog(self._controller, self).exec()

    def _build_autolabel_task(self) -> AutoLabelTask:
        return AutoLabelTask(
            task_type="segmentation",
            accent=_ACCENT,
            title="Segmentation",
            description=_AUTOLABEL_DESCRIPTION,
            default_input_size=autolabel_segmentation.DEFAULT_SIZE,
            build_predictor=autolabel_segmentation.build_predictor,
            apply_predictions=autolabel_segmentation.apply_predictions,
            needs_normalize_toggle=True,
            duplicate_warning=_AUTOLABEL_DUPLICATE_WARNING,
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

    def _on_delete_object(self, annotation_id: str) -> None:
        if self._current_image_id:
            self._controller.remove_annotation(self._current_image_id, annotation_id)
            self.canvas.remove_item(annotation_id)

    def _on_change_object_class(self, annotation_id: str, class_id: str) -> None:
        if self._current_image_id:
            self._controller.update_annotation(self._current_image_id, annotation_id, class_id=class_id)
            self.canvas.update_item_class(annotation_id, class_id)

    def _on_classes_changed(self) -> None:
        self.canvas.refresh_class_styles()
        self.canvas.load_image(self._current_image_id)

    def _export_from_neighbor(self, delta: int) -> None:
        if self._current_image_id is None:
            self.status_label.setText("Select an image first.")
            return
        neighbor_id = self.grid.neighbor_image_id(delta)
        direction = "previous" if delta < 0 else "next"
        if neighbor_id is None:
            self.status_label.setText(f"There's no {direction} image.")
            return

        project = self._controller.project
        neighbor = project.find_image(neighbor_id)
        current = project.find_image(self._current_image_id)
        if neighbor is None or not neighbor.annotations:
            self.status_label.setText(f"The {direction} image has no shapes to copy.")
            return

        if current is not None and current.annotations:
            reply = QMessageBox.question(
                self,
                f"Import from {direction.capitalize()}",
                f"This image already has {len(current.annotations)} shape(s). Replace them with the "
                f"{len(neighbor.annotations)} shape(s) from the {direction} image?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        count = self._controller.copy_annotations(neighbor_id, self._current_image_id)
        self.canvas.load_image(self._current_image_id)
        self.status_label.setText(f"Copied {count} shape(s) from the {direction} image.")
