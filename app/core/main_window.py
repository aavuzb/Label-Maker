"""Application shell: top toolbar, landing page, and task workspaces."""

from __future__ import annotations

from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QSizePolicy, QToolBar, QWidget

from app.core.help_dialog import HelpDialog
from app.features.classification.workspace import ClassificationWorkspace
from app.features.detection.workspace import DetectionWorkspace
from app.features.project.dialogs import OpenProjectDialog, prompt_new_project
from app.features.project.landing_page import LandingPage
from app.features.project.manager import ProjectController
from app.features.project.models import TASK_CLASSIFICATION, TASK_DETECTION, TASK_SEGMENTATION
from app.features.segmentation.workspace import SegmentationWorkspace
from app.shared.paths import ensure_app_dirs
from app.shared.theme import FS_BODY, TEXT_PRIMARY, app_icon_pixmap

_WORKSPACE_CLASSES = {
    TASK_CLASSIFICATION: ClassificationWorkspace,
    TASK_DETECTION: DetectionWorkspace,
    TASK_SEGMENTATION: SegmentationWorkspace,
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LabelMaker")
        self.resize(1200, 800)
        ensure_app_dirs()

        self.controller = ProjectController()
        self._workspace = None

        self._build_toolbar()
        self._build_shortcuts()
        self._show_landing_page()
        self.statusBar().showMessage("Ready")

        self.controller.project_opened.connect(self._on_project_opened)
        self.controller.project_closed.connect(self._show_landing_page)
        self.controller.images_added.connect(
            lambda added: self.statusBar().showMessage(f"Imported {len(added)} image(s)", 4000)
        )

    # -- UI construction -----------------------------------------------

    def _build_toolbar(self) -> None:
        self.toolbar = QToolBar("Main")
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        self.toolbar.addWidget(self._build_brand_widget())

        # Home only makes sense once a project is open; Help is always
        # available (a brand-new user needs it most on the landing page,
        # before picking a task at all), so it's pinned to the far right.
        self._home_action = self.toolbar.addAction("Home", self._go_home)
        self._home_action.setVisible(False)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        self.toolbar.addAction("Help", self._show_help)

    def _build_brand_widget(self) -> QWidget:
        """Small logo + wordmark pinned to the toolbar's left edge — shown
        on every page (landing page and all three workspaces alike), purely
        cosmetic, no effect on the Home/Help actions around it."""
        icon_label = QLabel()
        icon_label.setPixmap(app_icon_pixmap(20))
        icon_label.setStyleSheet("background: transparent; border: none;")

        name_label = QLabel("LabelMaker")
        name_label.setStyleSheet(
            f"background: transparent; border: none; font-weight: 700; font-size: {FS_BODY}px; color: {TEXT_PRIMARY};"
        )

        brand = QWidget()
        layout = QHBoxLayout(brand)
        layout.setContentsMargins(6, 0, 16, 0)
        layout.setSpacing(8)
        layout.addWidget(icon_label)
        layout.addWidget(name_label)
        return brand

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+N"), self, activated=self._go_home)
        QShortcut(QKeySequence.Open, self, activated=self._open_project)
        QShortcut(QKeySequence.HelpContents, self, activated=self._show_help)

    def _show_landing_page(self) -> None:
        landing = LandingPage()
        landing.task_selected.connect(self._new_project)
        landing.open_project_requested.connect(self._open_project)
        self.setCentralWidget(landing)
        self._workspace = None
        self._home_action.setVisible(False)
        self._update_title()

    def _show_help(self) -> None:
        HelpDialog(self).exec()

    # -- project lifecycle -----------------------------------------------

    def _new_project(self, task_type: str) -> None:
        project = prompt_new_project(self, task_type)
        if project is not None:
            self.controller.set_project(project)

    def _open_project(self) -> None:
        dialog = OpenProjectDialog(self)
        if dialog.exec():
            project_dir = dialog.selected_project_dir()
            if project_dir is not None:
                self.controller.open_project(project_dir)

    def _go_home(self) -> None:
        self.controller.close_project()

    def _on_project_opened(self) -> None:
        project = self.controller.project
        assert project is not None

        self._workspace = _WORKSPACE_CLASSES[project.task_type](self.controller)
        self.setCentralWidget(self._workspace)
        self._home_action.setVisible(True)
        self._update_title()

    def _update_title(self) -> None:
        project = self.controller.project
        self.setWindowTitle("LabelMaker" if project is None else f"LabelMaker — {project.name}")
