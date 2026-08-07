"""LabelMaker entry point."""

from __future__ import annotations

from app.shared.autolabel.cpu_budget import apply_cpu_thread_budget
from app.shared.autolabel.gpu import apply_gpu_library_path

# Must run before any other import in this file (or anything they pull in)
# ever touches numpy/scipy/torch/cv2 — see cpu_budget.py's docstring for why
# this ordering is load-bearing, not just tidiness.
apply_cpu_thread_budget()
# Same reasoning, different env var — see apply_gpu_library_path's
# docstring for why onnxruntime's GPU support needs this fixed before
# onnxruntime itself is ever imported anywhere in the process.
apply_gpu_library_path()

import sys  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.main_window import MainWindow  # noqa: E402
from app.shared.theme import APP_STYLESHEET, app_font, app_icon, build_palette  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LabelMaker")
    # Force Fusion: native/platform styles on some desktops (GTK themes,
    # KDE Breeze, etc.) partially ignore stylesheet borders on buttons.
    # Fusion always fully honors QSS, so the look is consistent everywhere.
    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setFont(app_font())
    app.setStyleSheet(APP_STYLESHEET)
    app.setWindowIcon(app_icon())

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
