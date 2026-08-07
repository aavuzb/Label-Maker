"""Filter selector: All / Unlabeled / By Class."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from app.shared.labeling.image_model import ImageFilterProxyModel
from app.features.project.manager import ProjectController


class FilterBar(QWidget):
    filter_changed = Signal(str, object)  # mode, class_id | None

    def __init__(self, controller: ProjectController, default_mode: str = ImageFilterProxyModel.FILTER_UNLABELED) -> None:
        super().__init__()
        self._controller = controller

        self.mode_combo = QComboBox()
        # "Unlabeled" first: for classification, the working queue empties
        # out as images get assigned, instead of leaving them visible mixed
        # in with "All". Detection/segmentation pass default_mode=FILTER_ALL
        # instead — annotating an image shouldn't make it vanish mid-edit.
        self.mode_combo.addItem("Unlabeled", ImageFilterProxyModel.FILTER_UNLABELED)
        self.mode_combo.addItem("All", ImageFilterProxyModel.FILTER_ALL)
        self.mode_combo.addItem("By Class", ImageFilterProxyModel.FILTER_CLASS)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(default_mode))
        self.mode_combo.currentIndexChanged.connect(self._emit_changed)

        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._emit_changed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Filter:"))
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.class_combo)
        layout.addStretch()

        controller.project_opened.connect(self._refresh_classes)
        controller.classes_changed.connect(self._refresh_classes)
        self._refresh_classes()

    def _refresh_classes(self) -> None:
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        project = self._controller.project
        if project is not None:
            for label in project.classes:
                self.class_combo.addItem(label.name, label.id)
        self.class_combo.blockSignals(False)
        is_class_mode = self.mode_combo.currentData() == ImageFilterProxyModel.FILTER_CLASS
        self.class_combo.setVisible(is_class_mode)
        self._emit_changed()

    def _emit_changed(self) -> None:
        mode = self.mode_combo.currentData()
        self.class_combo.setVisible(mode == ImageFilterProxyModel.FILTER_CLASS)
        class_id = self.class_combo.currentData() if mode == ImageFilterProxyModel.FILTER_CLASS else None
        self.filter_changed.emit(mode, class_id)
