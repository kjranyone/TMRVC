"""QApplication initialization and style setup."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from tmrvc_gui.main_window import MainWindow

_RESOURCES_DIR = Path(__file__).parent / "resources"


def run_app() -> None:
    """Create QApplication, apply stylesheet, and show the main window."""
    app = QApplication(sys.argv)
    app.setApplicationName("TMRVC Research Studio")
    app.setOrganizationName("TMRVC")

    # Apply stylesheet
    qss_path = _RESOURCES_DIR / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
