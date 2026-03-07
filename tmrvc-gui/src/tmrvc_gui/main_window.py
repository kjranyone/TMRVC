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

from tmrvc_gui.pages.codec_train import CodecTrainPage
from tmrvc_gui.pages.data_prep import DataPrepPage
from tmrvc_gui.pages.enrollment import EnrollmentPage
from tmrvc_gui.pages.evaluation import EvaluationPage
from tmrvc_gui.pages.onnx_export import OnnxExportPage
from tmrvc_gui.pages.realtime_demo import RealtimeDemoPage
from tmrvc_gui.pages.remote_client import RemoteClientPage
from tmrvc_gui.pages.script import ScriptPage
from tmrvc_gui.pages.server import ServerPage
from tmrvc_gui.pages.style_editor import StyleEditorPage
from tmrvc_gui.pages.token_train import TokenTrainPage
from tmrvc_gui.pages.tts import TTSPage
from tmrvc_gui.pages.curation import CurationPage
from tmrvc_gui.pages.admin_dashboard import AdminDashboardPage

_TABS: list[tuple[str, type[QWidget]]] = [
    ("Remote Client", RemoteClientPage),
    ("Data Prep", DataPrepPage),
    ("Codec Training", CodecTrainPage),
    ("UCLM Training", TokenTrainPage),
    ("Evaluation", EvaluationPage),
    ("Speaker Enrollment", EnrollmentPage),
    ("Realtime VC (UCLM)", RealtimeDemoPage),
    ("ONNX Export", OnnxExportPage),
    ("TTS (UCLM)", TTSPage),
    ("Batch Script", ScriptPage),
    ("Physical Style Editor", StyleEditorPage),
    ("Inference Server", ServerPage),
    ("Curation Auditor", CurationPage),
    ("System Admin", AdminDashboardPage),
]


class MainWindow(QMainWindow):
    """Application main window with sidebar + stacked content area."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TMRVC Research Studio")
        self.resize(1280, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(180)
        self._sidebar.setObjectName("sidebar")
        layout.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, stretch=1)

        for label, page_cls in _TABS:
            item = QListWidgetItem(label)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self._sidebar.addItem(item)
            self._stack.addWidget(page_cls())

        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._sidebar.setCurrentRow(0)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")
