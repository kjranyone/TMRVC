"""ServerPage: TTS server management, monitoring, and API testing."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ServerPage(QWidget):
    """Manage FastAPI TTS server: start/stop, character registration, API testing."""

    _log_signal = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._setup_ui()
        self._log_signal.connect(self._append_log)

        # Poll server status every 2 seconds
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_status)
        self._timer.start(2000)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Server controls ---
        ctrl_group = QGroupBox("Server")
        ctrl_layout = QVBoxLayout(ctrl_group)

        # Config row
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("TTS checkpoint:"))
        self._tts_ckpt = QLineEdit()
        self._tts_ckpt.setPlaceholderText("checkpoints/tts_best.pt")
        cfg_row.addWidget(self._tts_ckpt, stretch=1)
        btn_browse_tts = QPushButton("Browse...")
        btn_browse_tts.clicked.connect(
            lambda: self._browse_file(self._tts_ckpt, "PyTorch Checkpoint (*.pt)"),
        )
        cfg_row.addWidget(btn_browse_tts)
        ctrl_layout.addLayout(cfg_row)

        vc_row = QHBoxLayout()
        vc_row.addWidget(QLabel("VC checkpoint:"))
        self._vc_ckpt = QLineEdit()
        self._vc_ckpt.setPlaceholderText("checkpoints/distill/best.pt (optional)")
        vc_row.addWidget(self._vc_ckpt, stretch=1)
        btn_browse_vc = QPushButton("Browse...")
        btn_browse_vc.clicked.connect(
            lambda: self._browse_file(self._vc_ckpt, "PyTorch Checkpoint (*.pt)"),
        )
        vc_row.addWidget(btn_browse_vc)
        ctrl_layout.addLayout(vc_row)

        # Host/port
        net_row = QHBoxLayout()
        net_row.addWidget(QLabel("Host:"))
        self._host_edit = QLineEdit("127.0.0.1")
        self._host_edit.setFixedWidth(120)
        net_row.addWidget(self._host_edit)
        net_row.addWidget(QLabel("Port:"))
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(8000)
        net_row.addWidget(self._port_spin)
        net_row.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["xpu", "cpu", "cuda"])
        net_row.addWidget(self._device_combo)
        net_row.addStretch()
        ctrl_layout.addLayout(net_row)

        # Start/Stop buttons
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("Start Server")
        self._btn_start.clicked.connect(self._on_start)
        btn_row.addWidget(self._btn_start)
        self._btn_stop = QPushButton("Stop Server")
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_stop.setEnabled(False)
        btn_row.addWidget(self._btn_stop)
        self._status_label = QLabel("Status: Stopped")
        btn_row.addWidget(self._status_label)
        btn_row.addStretch()
        ctrl_layout.addLayout(btn_row)

        layout.addWidget(ctrl_group)

        # --- API Testing ---
        test_group = QGroupBox("API Test")
        test_layout = QVBoxLayout(test_group)

        text_row = QHBoxLayout()
        text_row.addWidget(QLabel("Text:"))
        self._test_text = QLineEdit()
        self._test_text.setText("Hello, this is a test.")
        text_row.addWidget(self._test_text, stretch=1)
        test_layout.addLayout(text_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Emotion:"))
        self._test_emotion = QComboBox()
        self._test_emotion.addItems([
            "neutral", "happy", "sad", "angry", "fearful", "surprised",
            "disgusted", "bored", "excited", "tender", "sarcastic", "whisper",
        ])
        param_row.addWidget(self._test_emotion)
        param_row.addWidget(QLabel("Endpoint:"))
        self._test_endpoint = QComboBox()
        self._test_endpoint.addItems(["/tts", "/tts/stream", "/health"])
        param_row.addWidget(self._test_endpoint)
        param_row.addStretch()
        test_layout.addLayout(param_row)

        test_btn_row = QHBoxLayout()
        btn_send = QPushButton("Send Request")
        btn_send.clicked.connect(self._on_send_request)
        test_btn_row.addWidget(btn_send)
        test_btn_row.addStretch()
        test_layout.addLayout(test_btn_row)

        self._response_log = QTextEdit()
        self._response_log.setReadOnly(True)
        self._response_log.setMaximumHeight(120)
        test_layout.addWidget(self._response_log)
        layout.addWidget(test_group)

        # --- Character Management ---
        char_group = QGroupBox("Characters")
        char_layout = QHBoxLayout(char_group)
        self._char_list = QTextEdit()
        self._char_list.setReadOnly(True)
        self._char_list.setPlaceholderText("Start server to see registered characters")
        char_layout.addWidget(self._char_list, stretch=1)

        char_btn_col = QVBoxLayout()
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._on_refresh_characters)
        char_btn_col.addWidget(btn_refresh)
        btn_register = QPushButton("Register...")
        btn_register.clicked.connect(self._on_register_character)
        char_btn_col.addWidget(btn_register)
        char_btn_col.addStretch()
        char_layout.addLayout(char_btn_col)

        layout.addWidget(char_group)

        # --- Server log ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        layout.addWidget(QLabel("Server Log"))
        layout.addWidget(self._log)

    def _browse_file(self, line_edit: QLineEdit, filter_str: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if path:
            line_edit.setText(path)

    def _append_log(self, msg: str) -> None:
        self._log.append(msg)

    # --- Server lifecycle ---

    def _on_start(self) -> None:
        tts_ckpt = self._tts_ckpt.text().strip()
        if not tts_ckpt:
            QMessageBox.warning(self, "Error", "TTS checkpoint path is required")
            return

        cmd = [
            sys.executable, "-m", "tmrvc_serve",
            "--tts-checkpoint", tts_ckpt,
            "--host", self._host_edit.text(),
            "--port", str(self._port_spin.value()),
            "--device", self._device_combo.currentText(),
        ]
        vc_ckpt = self._vc_ckpt.text().strip()
        if vc_ckpt:
            cmd.extend(["--vc-checkpoint", vc_ckpt])

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start server: {e}")
            return

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status_label.setText("Status: Starting...")
        self._append_log("Server starting...")

        # Read output in background thread
        self._reader_thread = threading.Thread(
            target=self._read_output, daemon=True,
        )
        self._reader_thread.start()

    def _on_stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process = None
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status_label.setText("Status: Stopped")
        self._append_log("Server stopped")

    def _read_output(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self._log_signal.emit(line)
        self._log_signal.emit("Server process exited")

    def _update_status(self) -> None:
        if self._process is not None:
            if self._process.poll() is not None:
                self._status_label.setText("Status: Exited")
                self._btn_start.setEnabled(True)
                self._btn_stop.setEnabled(False)
                self._process = None
            else:
                self._status_label.setText("Status: Running")

    # --- API testing ---

    def _on_send_request(self) -> None:
        endpoint = self._test_endpoint.currentText()
        host = self._host_edit.text()
        port = self._port_spin.value()
        base_url = f"http://{host}:{port}"

        try:
            import urllib.request
            import urllib.error

            if endpoint == "/health":
                url = f"{base_url}/health"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    body = resp.read().decode("utf-8")
                self._response_log.setText(body)
            elif endpoint == "/tts":
                url = f"{base_url}/tts"
                payload = json.dumps({
                    "text": self._test_text.text(),
                    "emotion": self._test_emotion.currentText(),
                }).encode("utf-8")
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self._response_log.setText(
                    f"Duration: {body.get('duration_sec', '?')}s\n"
                    f"Style: {json.dumps(body.get('style_used', {}), indent=2)}\n"
                    f"Audio: {len(body.get('audio_base64', ''))} chars (base64)"
                )
            else:
                self._response_log.setText(f"Streaming endpoint not supported in GUI test")
        except Exception as e:
            self._response_log.setText(f"Error: {e}")

    def _on_refresh_characters(self) -> None:
        host = self._host_edit.text()
        port = self._port_spin.value()
        try:
            import urllib.request
            url = f"http://{host}:{port}/characters"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            self._char_list.setText(json.dumps(body, indent=2, ensure_ascii=False))
        except Exception as e:
            self._char_list.setText(f"Error: {e}")

    def _on_register_character(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Character or Speaker File", "",
            "Character/Speaker Files (*.tmrvc_character *.tmrvc_speaker);;All Files (*)",
        )
        if not path:
            return
        self._append_log(f"Register character from: {path}")
        # Registration would be done via POST /characters API
        self._response_log.setText(f"TODO: POST /characters with file: {path}")
