"""MainWindow: sidebar navigation + stacked pages + status bar."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from tmrvc_gui.pages.data_prep import DataPrepPage
from tmrvc_gui.pages.distillation import DistillationPage
from tmrvc_gui.pages.enrollment import EnrollmentPage
from tmrvc_gui.pages.evaluation import EvaluationPage
from tmrvc_gui.pages.onnx_export import OnnxExportPage
from tmrvc_gui.pages.realtime_demo import RealtimeDemoPage
from tmrvc_gui.pages.teacher_train import TeacherTrainPage

_TABS: list[tuple[str, type[QWidget]]] = [
    ("Data Prep", DataPrepPage),
    ("Teacher Training", TeacherTrainPage),
    ("Distillation", DistillationPage),
    ("Evaluation", EvaluationPage),
    ("Speaker Enrollment", EnrollmentPage),
    ("Realtime Demo", RealtimeDemoPage),
    ("ONNX Export", OnnxExportPage),
]


class MainWindow(QMainWindow):
    """Application main window with sidebar + stacked content area."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TMRVC Research Studio")
        self.resize(1280, 800)

        # --- Central widget ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Sidebar ---
        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(180)
        self._sidebar.setObjectName("sidebar")
        layout.addWidget(self._sidebar)

        # --- Stacked pages ---
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, stretch=1)

        # --- Populate tabs ---
        for label, page_cls in _TABS:
            item = QListWidgetItem(label)
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._sidebar.addItem(item)
            self._stack.addWidget(page_cls())

        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._sidebar.setCurrentRow(0)

        # --- Status bar ---
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("Ready")
        self._model_label = QLabel("Model: (none)")
        self._latency_label = QLabel("Latency: --")

        self._status_bar.addWidget(self._status_label, stretch=1)
        self._status_bar.addPermanentWidget(self._model_label)
        self._status_bar.addPermanentWidget(self._latency_label)

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_model_info(self, text: str) -> None:
        self._model_label.setText(f"Model: {text}")

    def set_latency_info(self, text: str) -> None:
        self._latency_label.setText(f"Latency: {text}")
