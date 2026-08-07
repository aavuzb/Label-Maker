"""Dialog for creating or editing a single class label."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
)


class ClassDialog(QDialog):
    def __init__(self, name: str = "", color: str = "#3498db", shortcut: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Class")
        self.setMinimumWidth(340)
        self._color = color

        self.name_edit = QLineEdit(name)
        self.shortcut_edit = QLineEdit(shortcut)
        self.shortcut_edit.setMaxLength(1)
        self.shortcut_edit.setPlaceholderText("e.g. 1")

        self.color_button = QPushButton()
        self.color_button.clicked.connect(self._pick_color)
        self._update_color_button()

        form = QFormLayout()
        form.setContentsMargins(18, 18, 18, 16)
        form.setSpacing(12)
        form.addRow("Name", self.name_edit)
        form.addRow("Color", self.color_button)
        form.addRow("Shortcut", self.shortcut_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self.setLayout(form)
        form.addRow(buttons)

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._color), self, "Class Color")
        if color.isValid():
            self._color = color.name()
            self._update_color_button()

    def _update_color_button(self) -> None:
        self.color_button.setText(self._color)
        self.color_button.setStyleSheet(f"background-color: {self._color};")

    def values(self) -> tuple[str, str, str]:
        return self.name_edit.text().strip(), self._color, self.shortcut_edit.text().strip()
