"""Dialog for exporting a classification project to ImageFolder / CSV."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from app.features.classification.export import FORMAT_IMAGEFOLDER, FORMAT_LABELS, export_dataset
from app.features.project.manager import ProjectController
from app.shared.labeling.panel_widgets import build_callout
from app.shared.theme import ACCENT, FS_CAPTION, primary_button_style

_FORMAT_HELP = {
    "imagefolder": (
        "{train,val,test}/{class name}/ folders of images — the layout torchvision's ImageFolder, Keras' "
        "flow_from_directory, and most other classifier training tools expect directly."
    ),
    "csv": (
        "images/{train,val,test}/ of images, plus a {split}.csv file per split (columns: filename, label) "
        "— for training code that reads labels from a manifest instead of folder structure."
    ),
}


class ExportDialog(QDialog):
    def __init__(self, controller: ProjectController, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle("Export Dataset")
        self.setMinimumWidth(460)

        self.format_combo = QComboBox()
        for value, label in FORMAT_LABELS.items():
            self.format_combo.addItem(label, value)
        self.format_combo.currentIndexChanged.connect(self._update_help)

        self.help_label = build_callout(_FORMAT_HELP[FORMAT_IMAGEFOLDER], ACCENT)

        self.train_spin = self._ratio_spin(70)
        self.val_spin = self._ratio_spin(20)
        self.test_spin = self._ratio_spin(10)
        for spin in (self.train_spin, self.val_spin, self.test_spin):
            spin.valueChanged.connect(self._update_split_total)

        split_row = QHBoxLayout()
        split_row.setSpacing(6)
        for text, spin in (("Train", self.train_spin), ("Val", self.val_spin), ("Test", self.test_spin)):
            split_row.addWidget(QLabel(text))
            split_row.addWidget(spin)
        split_row.addStretch()

        self.split_total_label = QLabel()
        self.split_total_label.setStyleSheet(f"font-size: {FS_CAPTION}px;")

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Choose an output folder…")
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, stretch=1)
        output_row.addWidget(browse_button)

        form = QFormLayout()
        form.setContentsMargins(20, 18, 20, 16)
        form.setSpacing(12)
        form.addRow("Format", self.format_combo)
        form.addRow(self.help_label)
        form.addRow("Split", split_row)
        form.addRow("", self.split_total_label)
        form.addRow("Output folder", output_row)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Export")
        self.buttons.button(QDialogButtonBox.Ok).setStyleSheet(primary_button_style(ACCENT))
        self.buttons.accepted.connect(self._do_export)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 16)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

        self._update_split_total()

    def _ratio_spin(self, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setSuffix("%")
        spin.setValue(value)
        return spin

    def _update_help(self) -> None:
        self.help_label.setText(_FORMAT_HELP[self.format_combo.currentData()])

    def _update_split_total(self) -> None:
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        ok = total == 100
        color = "#16a34a" if ok else "#dc2626"
        self.split_total_label.setText(f"Total: {total}%" if ok else f"Total: {total}% — must equal 100%")
        self.split_total_label.setStyleSheet(f"color: {color}; font-size: {FS_CAPTION}px;")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export Output Folder")
        if folder:
            self.output_edit.setText(folder)

    def _do_export(self) -> None:
        project = self._controller.project
        if project is None:
            return

        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "Export Dataset", "Choose an output folder first.")
            return
        if not project.classes:
            QMessageBox.warning(self, "Export Dataset", "Add at least one class before exporting.")
            return

        fmt = self.format_combo.currentData()
        out_dir = Path(output_text)
        try:
            result = export_dataset(
                project,
                out_dir,
                fmt,
                self.train_spin.value() / 100,
                self.val_spin.value() / 100,
                self.test_spin.value() / 100,
            )
        except OSError as exc:
            QMessageBox.critical(self, "Export Dataset", f"Export failed: {exc}")
            return

        if result.exported_count == 0:
            QMessageBox.warning(self, "Export Dataset", "Nothing was exported — no images have been assigned a class yet.")
            return

        split_summary = ", ".join(f"{split}: {count}" for split, count in result.split_counts.items())
        message = f"Exported {result.exported_count} image(s) ({split_summary})\nto {out_dir}"
        if result.skipped_unlabeled:
            message += f"\n\nSkipped {result.skipped_unlabeled} image(s) with no class assigned."
        if result.skipped_unreadable:
            message += f"\nSkipped {result.skipped_unreadable} image(s) that couldn't be read."
        QMessageBox.information(self, "Export Dataset", message)
        self.accept()
