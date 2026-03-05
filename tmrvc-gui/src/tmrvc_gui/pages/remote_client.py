"""Remote Client page: full client for a remote tmrvc-serve instance.

Covers all tmrvc-serve API endpoints:
- GET /health
- GET/POST /characters
- POST /tts, POST /tts/stream
- WS /ws/chat (WebSocket live TTS)
- WS /vc/stream (WebSocket real-time VC)
- GET /vc/stats
- POST /auth/token, /auth/refresh, /auth/logout
- GET /auth/me, /auth/usage, /auth/keys
- POST /auth/keys, DELETE /auth/keys/{prefix}
"""

from __future__ import annotations

import base64
import json
import struct
import threading
import urllib.error
import urllib.request

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

EMOTION_OPTIONS = [
    "neutral", "happy", "sad", "angry", "fearful", "surprised",
    "disgusted", "bored", "excited", "tender", "sarcastic", "whisper",
]

STYLE_PRESETS = ["default", "asmr_soft", "asmr_intimate"]


class RemoteClientPage(QWidget):
    """Full client for a remote tmrvc-serve instance."""

    _log_signal = Signal(str)
    _conn_status_signal = Signal(str)
    _characters_signal = Signal(str)
    _result_signal = Signal(bool, str)
    _ws_audio_signal = Signal(bytes)
    _ws_status_signal = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_audio_bytes: bytes | None = None
        self._ws_chat = None  # websocket connection
        self._ws_vc = None
        self._ws_chat_running = False
        self._ws_vc_running = False
        self._auth_token: str | None = None
        self._setup_ui()
        self._log_signal.connect(self._append_log)
        self._conn_status_signal.connect(lambda s: self._status_label.setText(s))
        self._characters_signal.connect(self._on_characters_loaded)
        self._result_signal.connect(self._on_tts_result)
        self._ws_status_signal.connect(self._on_ws_status)
        self._ws_audio_signal.connect(self._on_ws_audio_received)

        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._poll_health)
        self._health_timer.start(5000)

    # ==================================================================
    # UI
    # ==================================================================

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # --- Connection bar (always visible) ---
        conn_group = QGroupBox("Connection")
        conn_layout = QVBoxLayout(conn_group)

        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Host:"))
        self._host_edit = QLineEdit("127.0.0.1")
        self._host_edit.setFixedWidth(180)
        conn_row.addWidget(self._host_edit)
        conn_row.addWidget(QLabel("Port:"))
        self._port_edit = QLineEdit("8000")
        self._port_edit.setFixedWidth(80)
        conn_row.addWidget(self._port_edit)
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.clicked.connect(self._on_connect)
        conn_row.addWidget(self._btn_connect)
        self._status_label = QLabel("Disconnected")
        conn_row.addWidget(self._status_label)
        conn_row.addStretch()
        conn_layout.addLayout(conn_row)
        layout.addWidget(conn_group)

        # --- Sub-tabs ---
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tts_tab(), "TTS")
        self._tabs.addTab(self._build_ws_chat_tab(), "WS Chat")
        self._tabs.addTab(self._build_vc_tab(), "VC Stream")
        self._tabs.addTab(self._build_characters_tab(), "Characters")
        self._tabs.addTab(self._build_auth_tab(), "Auth")
        layout.addWidget(self._tabs, stretch=1)

        # --- Log (always visible) ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(150)
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self._log)

    # --- TTS tab ---
    def _build_tts_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        char_row = QHBoxLayout()
        char_row.addWidget(QLabel("Character:"))
        self._tts_char_combo = QComboBox()
        self._tts_char_combo.setMinimumWidth(160)
        char_row.addWidget(self._tts_char_combo)
        char_row.addWidget(QLabel("Emotion:"))
        self._tts_emotion = QComboBox()
        self._tts_emotion.addItems(EMOTION_OPTIONS)
        char_row.addWidget(self._tts_emotion)
        char_row.addWidget(QLabel("Preset:"))
        self._tts_preset = QComboBox()
        self._tts_preset.addItems(STYLE_PRESETS)
        char_row.addWidget(self._tts_preset)
        char_row.addWidget(QLabel("Speed:"))
        self._tts_speed = QDoubleSpinBox()
        self._tts_speed.setRange(0.5, 2.0)
        self._tts_speed.setSingleStep(0.1)
        self._tts_speed.setValue(1.0)
        char_row.addWidget(self._tts_speed)
        char_row.addStretch()
        layout.addLayout(char_row)

        self._tts_text = QTextEdit()
        self._tts_text.setPlaceholderText("Enter text to synthesize...")
        self._tts_text.setMaximumHeight(80)
        layout.addWidget(self._tts_text)

        hint_row = QHBoxLayout()
        hint_row.addWidget(QLabel("Hint:"))
        self._tts_hint = QLineEdit()
        self._tts_hint.setPlaceholderText("(optional) acting direction")
        hint_row.addWidget(self._tts_hint, stretch=1)
        hint_row.addWidget(QLabel("Situation:"))
        self._tts_situation = QLineEdit()
        self._tts_situation.setPlaceholderText("(optional) scene description")
        hint_row.addWidget(self._tts_situation, stretch=1)
        layout.addLayout(hint_row)

        btn_row = QHBoxLayout()
        self._btn_generate = QPushButton("Generate (POST /tts)")
        self._btn_generate.clicked.connect(self._on_generate)
        btn_row.addWidget(self._btn_generate)
        self._btn_stream = QPushButton("Stream (POST /tts/stream)")
        self._btn_stream.clicked.connect(self._on_stream)
        btn_row.addWidget(self._btn_stream)
        self._btn_play = QPushButton("Play")
        self._btn_play.setEnabled(False)
        self._btn_play.clicked.connect(self._on_play)
        btn_row.addWidget(self._btn_play)
        self._btn_save = QPushButton("Save WAV...")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(self._btn_save)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._tts_response = QTextEdit()
        self._tts_response.setReadOnly(True)
        self._tts_response.setPlaceholderText("Response details will appear here")
        layout.addWidget(self._tts_response)
        return w

    # --- WS Chat tab ---
    def _build_ws_chat_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Character:"))
        self._ws_char_combo = QComboBox()
        self._ws_char_combo.setMinimumWidth(160)
        cfg_row.addWidget(self._ws_char_combo)
        cfg_row.addWidget(QLabel("Preset:"))
        self._ws_preset = QComboBox()
        self._ws_preset.addItems(STYLE_PRESETS)
        cfg_row.addWidget(self._ws_preset)
        cfg_row.addWidget(QLabel("Speed:"))
        self._ws_speed = QDoubleSpinBox()
        self._ws_speed.setRange(0.5, 2.0)
        self._ws_speed.setSingleStep(0.1)
        self._ws_speed.setValue(1.0)
        cfg_row.addWidget(self._ws_speed)
        cfg_row.addStretch()
        layout.addLayout(cfg_row)

        ctrl_row = QHBoxLayout()
        self._btn_ws_connect = QPushButton("Connect WS")
        self._btn_ws_connect.clicked.connect(self._on_ws_chat_connect)
        ctrl_row.addWidget(self._btn_ws_connect)
        self._btn_ws_disconnect = QPushButton("Disconnect")
        self._btn_ws_disconnect.setEnabled(False)
        self._btn_ws_disconnect.clicked.connect(self._on_ws_chat_disconnect)
        ctrl_row.addWidget(self._btn_ws_disconnect)
        self._btn_ws_cancel = QPushButton("Cancel Queue")
        self._btn_ws_cancel.setEnabled(False)
        self._btn_ws_cancel.clicked.connect(self._on_ws_chat_cancel)
        ctrl_row.addWidget(self._btn_ws_cancel)
        self._ws_status_label = QLabel("WS: Disconnected")
        ctrl_row.addWidget(self._ws_status_label)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        speak_row = QHBoxLayout()
        self._ws_text = QLineEdit()
        self._ws_text.setPlaceholderText("Type message to speak...")
        self._ws_text.returnPressed.connect(self._on_ws_chat_speak)
        speak_row.addWidget(self._ws_text, stretch=1)
        speak_row.addWidget(QLabel("Emotion:"))
        self._ws_emotion = QComboBox()
        self._ws_emotion.addItems(EMOTION_OPTIONS)
        speak_row.addWidget(self._ws_emotion)
        speak_row.addWidget(QLabel("Priority:"))
        self._ws_priority = QComboBox()
        self._ws_priority.addItems(["URGENT (0)", "NORMAL (1)", "LOW (2)"])
        self._ws_priority.setCurrentIndex(1)
        speak_row.addWidget(self._ws_priority)
        self._btn_ws_speak = QPushButton("Speak")
        self._btn_ws_speak.clicked.connect(self._on_ws_chat_speak)
        speak_row.addWidget(self._btn_ws_speak)
        layout.addLayout(speak_row)

        self._ws_chat_log = QTextEdit()
        self._ws_chat_log.setReadOnly(True)
        self._ws_chat_log.setPlaceholderText("WebSocket messages will appear here")
        layout.addWidget(self._ws_chat_log)
        return w

    # --- VC Stream tab ---
    def _build_vc_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        layout.addWidget(QLabel(
            "Real-time VC via WebSocket (WS /vc/stream).\n"
            "Requires API key for authentication."
        ))

        api_row = QHBoxLayout()
        api_row.addWidget(QLabel("API Key:"))
        self._vc_api_key = QLineEdit()
        self._vc_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._vc_api_key.setPlaceholderText("tmrvc_...")
        api_row.addWidget(self._vc_api_key, stretch=1)
        layout.addLayout(api_row)

        spk_row = QHBoxLayout()
        spk_row.addWidget(QLabel("Speaker file:"))
        self._vc_speaker_edit = QLineEdit()
        self._vc_speaker_edit.setPlaceholderText("path/to/speaker.tmrvc_speaker or .npy")
        spk_row.addWidget(self._vc_speaker_edit, stretch=1)
        btn_browse_spk = QPushButton("Browse...")
        btn_browse_spk.clicked.connect(self._on_vc_browse_speaker)
        spk_row.addWidget(btn_browse_spk)
        layout.addLayout(spk_row)

        ctrl_row = QHBoxLayout()
        self._btn_vc_connect = QPushButton("Connect VC Stream")
        self._btn_vc_connect.clicked.connect(self._on_vc_connect)
        ctrl_row.addWidget(self._btn_vc_connect)
        self._btn_vc_disconnect = QPushButton("Disconnect")
        self._btn_vc_disconnect.setEnabled(False)
        self._btn_vc_disconnect.clicked.connect(self._on_vc_disconnect)
        ctrl_row.addWidget(self._btn_vc_disconnect)
        self._btn_vc_stats = QPushButton("GET /vc/stats")
        self._btn_vc_stats.clicked.connect(self._on_vc_stats)
        ctrl_row.addWidget(self._btn_vc_stats)
        self._vc_status_label = QLabel("VC: Disconnected")
        ctrl_row.addWidget(self._vc_status_label)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        self._vc_log = QTextEdit()
        self._vc_log.setReadOnly(True)
        self._vc_log.setPlaceholderText("VC streaming status will appear here")
        layout.addWidget(self._vc_log)
        return w

    # --- Characters tab ---
    def _build_characters_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        top_row = QHBoxLayout()
        self._btn_refresh_chars = QPushButton("Refresh (GET /characters)")
        self._btn_refresh_chars.clicked.connect(self._on_refresh_characters)
        top_row.addWidget(self._btn_refresh_chars)
        top_row.addStretch()
        layout.addLayout(top_row)

        self._char_list_view = QTextEdit()
        self._char_list_view.setReadOnly(True)
        self._char_list_view.setPlaceholderText("Character list")
        layout.addWidget(self._char_list_view)

        # Register new character
        reg_group = QGroupBox("Register Character (POST /characters)")
        reg_form = QFormLayout(reg_group)
        self._reg_id = QLineEdit()
        self._reg_id.setPlaceholderText("character_id")
        reg_form.addRow("ID:", self._reg_id)
        self._reg_name = QLineEdit()
        reg_form.addRow("Name:", self._reg_name)
        self._reg_personality = QLineEdit()
        reg_form.addRow("Personality:", self._reg_personality)
        self._reg_voice_desc = QLineEdit()
        reg_form.addRow("Voice Description:", self._reg_voice_desc)
        self._reg_language = QComboBox()
        self._reg_language.addItems(["ja", "en", "zh", "ko"])
        reg_form.addRow("Language:", self._reg_language)
        spk_row = QHBoxLayout()
        self._reg_speaker_file = QLineEdit()
        self._reg_speaker_file.setPlaceholderText("(optional) path on server")
        spk_row.addWidget(self._reg_speaker_file)
        reg_form.addRow("Speaker file:", spk_row)
        self._btn_register = QPushButton("Register")
        self._btn_register.clicked.connect(self._on_register_character)
        reg_form.addRow(self._btn_register)
        layout.addWidget(reg_group)
        return w

    # --- Auth tab ---
    def _build_auth_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # Login
        login_group = QGroupBox("Login (POST /auth/token)")
        login_form = QFormLayout(login_group)
        self._auth_email = QLineEdit()
        self._auth_email.setPlaceholderText("admin@tmrvc.example.com")
        login_form.addRow("Email:", self._auth_email)
        self._auth_password = QLineEdit()
        self._auth_password.setEchoMode(QLineEdit.EchoMode.Password)
        login_form.addRow("Password:", self._auth_password)
        btn_row = QHBoxLayout()
        self._btn_login = QPushButton("Login")
        self._btn_login.clicked.connect(self._on_login)
        btn_row.addWidget(self._btn_login)
        self._btn_logout = QPushButton("Logout")
        self._btn_logout.clicked.connect(self._on_logout)
        btn_row.addWidget(self._btn_logout)
        self._auth_status = QLabel("Not authenticated")
        btn_row.addWidget(self._auth_status)
        btn_row.addStretch()
        login_form.addRow(btn_row)
        layout.addWidget(login_group)

        # Info buttons
        info_row = QHBoxLayout()
        btn_me = QPushButton("GET /auth/me")
        btn_me.clicked.connect(self._on_auth_me)
        info_row.addWidget(btn_me)
        btn_usage = QPushButton("GET /auth/usage")
        btn_usage.clicked.connect(self._on_auth_usage)
        info_row.addWidget(btn_usage)
        btn_keys = QPushButton("GET /auth/keys")
        btn_keys.clicked.connect(self._on_auth_keys)
        info_row.addWidget(btn_keys)
        info_row.addStretch()
        layout.addLayout(info_row)

        # Create API key
        key_group = QGroupBox("Create API Key (POST /auth/keys) — Admin only")
        key_form = QFormLayout(key_group)
        self._key_user_id = QLineEdit()
        key_form.addRow("User ID:", self._key_user_id)
        self._key_role = QComboBox()
        self._key_role.addItems(["free", "pro", "enterprise", "admin"])
        self._key_role.setCurrentIndex(1)
        key_form.addRow("Role:", self._key_role)
        self._key_expires = QSpinBox()
        self._key_expires.setRange(0, 365)
        self._key_expires.setValue(30)
        self._key_expires.setSpecialValueText("No expiry")
        key_form.addRow("Expires (days):", self._key_expires)
        self._btn_create_key = QPushButton("Create Key")
        self._btn_create_key.clicked.connect(self._on_create_key)
        key_form.addRow(self._btn_create_key)
        layout.addWidget(key_group)

        # Revoke key
        revoke_row = QHBoxLayout()
        revoke_row.addWidget(QLabel("Revoke key prefix:"))
        self._revoke_prefix = QLineEdit()
        self._revoke_prefix.setPlaceholderText("tmrvc_abc...")
        revoke_row.addWidget(self._revoke_prefix)
        self._btn_revoke = QPushButton("Revoke (DELETE)")
        self._btn_revoke.clicked.connect(self._on_revoke_key)
        revoke_row.addWidget(self._btn_revoke)
        revoke_row.addStretch()
        layout.addLayout(revoke_row)

        # Response display
        self._auth_response = QTextEdit()
        self._auth_response.setReadOnly(True)
        self._auth_response.setPlaceholderText("Auth API responses")
        layout.addWidget(self._auth_response)
        return w

    # ==================================================================
    # HTTP helpers
    # ==================================================================

    def _base_url(self) -> str:
        host = self._host_edit.text().strip() or "127.0.0.1"
        port = self._port_edit.text().strip() or "8000"
        return f"http://{host}:{port}"

    def _ws_url(self) -> str:
        host = self._host_edit.text().strip() or "127.0.0.1"
        port = self._port_edit.text().strip() or "8000"
        return f"ws://{host}:{port}"

    def _append_log(self, msg: str) -> None:
        self._log.append(msg)

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers

    def _http_get(self, path: str, timeout: float = 5) -> dict | list | None:
        url = f"{self._base_url()}{path}"
        req = urllib.request.Request(url, headers=self._auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post(self, path: str, body: dict, timeout: float = 30) -> dict:
        url = f"{self._base_url()}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_delete(self, path: str, timeout: float = 10) -> dict:
        url = f"{self._base_url()}{path}"
        req = urllib.request.Request(url, method="DELETE", headers=self._auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post_raw(self, path: str, body: dict, timeout: float = 60) -> bytes:
        """POST and return raw response bytes (for streaming endpoint)."""
        url = f"{self._base_url()}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _format_error(self, e: Exception) -> str:
        if isinstance(e, urllib.error.HTTPError):
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            return f"HTTP {e.code}: {body}"
        return str(e)

    # ==================================================================
    # Connection
    # ==================================================================

    def _on_connect(self) -> None:
        self._status_label.setText("Connecting...")
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self) -> None:
        try:
            health = self._http_get("/health")
            if not isinstance(health, dict) or "models_loaded" not in health:
                self._log_signal.emit(
                    "Connection failed: server responded but is not tmrvc-serve "
                    f"(got {json.dumps(health)[:200]})"
                )
                self._conn_status_signal.emit("Disconnected")
                return
            status = health.get("status", "?")
            models = health.get("models_loaded", False)
            n_chars = health.get("characters_count", 0)
            self._log_signal.emit(
                f"Connected: status={status}, models={models}, characters={n_chars}"
            )
            chars = self._http_get("/characters")
            self._characters_signal.emit(json.dumps(chars or []))
        except Exception as e:
            self._log_signal.emit(f"Connection failed: {self._format_error(e)}")
            self._conn_status_signal.emit("Disconnected")

    def _on_characters_loaded(self, chars_json: str) -> None:
        chars = json.loads(chars_json)
        for combo in (self._tts_char_combo, self._ws_char_combo):
            combo.clear()
            for c in chars:
                cid = c.get("id", "")
                name = c.get("name", cid)
                combo.addItem(f"{name} ({cid})", cid)
        if chars:
            self._status_label.setText(f"Connected ({len(chars)} characters)")
        else:
            self._status_label.setText("Connected (no characters)")

    def _poll_health(self) -> None:
        if self._status_label.text() == "Disconnected":
            return
        threading.Thread(target=self._health_worker, daemon=True).start()

    def _health_worker(self) -> None:
        try:
            health = self._http_get("/health", timeout=3)
            if (
                not isinstance(health, dict)
                or "models_loaded" not in health
                or health.get("status") != "ok"
            ):
                self._log_signal.emit("Health: not a valid tmrvc-serve response")
                self._conn_status_signal.emit("Disconnected")
        except Exception:
            self._log_signal.emit("Health check failed — server unreachable")
            self._conn_status_signal.emit("Disconnected")

    # ==================================================================
    # TTS (POST /tts)
    # ==================================================================

    def _build_tts_body(self) -> dict | None:
        text = self._tts_text.toPlainText().strip()
        if not text:
            self._append_log("ERROR: No text entered.")
            return None
        char_id = self._tts_char_combo.currentData()
        if not char_id:
            self._append_log("ERROR: No character selected.")
            return None
        body: dict = {
            "text": text,
            "character_id": char_id,
            "emotion": self._tts_emotion.currentText(),
            "style_preset": self._tts_preset.currentText(),
            "speed": self._tts_speed.value(),
        }
        hint = self._tts_hint.text().strip()
        if hint:
            body["hint"] = hint
        situation = self._tts_situation.text().strip()
        if situation:
            body["situation"] = situation
        return body

    def _on_generate(self) -> None:
        body = self._build_tts_body()
        if not body:
            return
        self._btn_generate.setEnabled(False)
        self._btn_stream.setEnabled(False)
        self._append_log(f"POST /tts: \"{body['text'][:60]}...\"")
        threading.Thread(target=self._generate_worker, args=(body,), daemon=True).start()

    def _generate_worker(self, body: dict) -> None:
        try:
            resp = self._http_post("/tts", body, timeout=60)
            audio_b64 = resp.get("audio_base64", "")
            duration = resp.get("duration_sec", 0)
            style_used = resp.get("style_used", {})
            self._last_audio_bytes = base64.b64decode(audio_b64)
            msg = f"Done: {duration:.2f}s"
            if style_used.get("emotion"):
                msg += f", emotion={style_used['emotion']}"
            self._result_signal.emit(True, msg)
        except Exception as e:
            self._result_signal.emit(False, self._format_error(e))

    # TTS Stream (POST /tts/stream)
    def _on_stream(self) -> None:
        body = self._build_tts_body()
        if not body:
            return
        body["chunk_duration_ms"] = 100
        self._btn_generate.setEnabled(False)
        self._btn_stream.setEnabled(False)
        self._append_log(f"POST /tts/stream: \"{body['text'][:60]}...\"")
        threading.Thread(target=self._stream_worker, args=(body,), daemon=True).start()

    def _stream_worker(self, body: dict) -> None:
        try:
            raw = self._http_post_raw("/tts/stream", body, timeout=60)
            self._last_audio_bytes = raw
            self._result_signal.emit(True, f"Stream received: {len(raw)} bytes")
        except Exception as e:
            self._result_signal.emit(False, self._format_error(e))

    def _on_tts_result(self, success: bool, message: str) -> None:
        self._btn_generate.setEnabled(True)
        self._btn_stream.setEnabled(True)
        self._append_log(message)
        if success:
            self._btn_play.setEnabled(True)
            self._btn_save.setEnabled(True)

    def _on_play(self) -> None:
        if not self._last_audio_bytes:
            return
        try:
            import io
            import soundfile as sf
            import sounddevice as sd
            audio, sr = sf.read(io.BytesIO(self._last_audio_bytes))
            sd.play(audio, sr)
            self._append_log("Playing...")
        except Exception as e:
            self._append_log(f"Playback error: {e}")

    def _on_save(self) -> None:
        if not self._last_audio_bytes:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "output.wav", "WAV Files (*.wav)",
        )
        if not path:
            return
        try:
            with open(path, "wb") as f:
                f.write(self._last_audio_bytes)
            self._append_log(f"Saved to {path}")
        except Exception as e:
            self._append_log(f"Save error: {e}")

    # ==================================================================
    # WebSocket Chat (WS /ws/chat)
    # ==================================================================

    def _on_ws_chat_connect(self) -> None:
        if self._ws_chat_running:
            return
        threading.Thread(target=self._ws_chat_worker, daemon=True).start()

    def _ws_chat_worker(self) -> None:
        try:
            import websockets.sync.client as ws_client
        except ImportError:
            self._log_signal.emit("ERROR: 'websockets' package required. pip install websockets")
            return

        url = f"{self._ws_url()}/ws/chat"
        self._ws_status_signal.emit("connecting")
        try:
            self._ws_chat = ws_client.connect(url)
            self._ws_chat_running = True
            self._ws_status_signal.emit("connected")

            # Send initial configure
            char_id = self._ws_char_combo.currentData()
            if char_id:
                cfg = {
                    "type": "configure",
                    "character_id": char_id,
                    "speed": self._ws_speed.value(),
                    "style_preset": self._ws_preset.currentText(),
                }
                self._ws_chat.send(json.dumps(cfg))

            # Read loop
            while self._ws_chat_running:
                try:
                    msg = self._ws_chat.recv(timeout=1.0)
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        msg_type = data.get("type", "")
                        if msg_type == "audio":
                            audio_b64 = data.get("data", "")
                            if audio_b64:
                                self._ws_audio_signal.emit(base64.b64decode(audio_b64))
                            if data.get("is_last"):
                                self._log_signal.emit(f"[WS] Audio seq={data.get('seq')} complete")
                        elif msg_type == "style":
                            self._log_signal.emit(
                                f"[WS] Style: emotion={data.get('emotion')}, "
                                f"VAD={data.get('vad')}"
                            )
                        elif msg_type == "queue_status":
                            self._log_signal.emit(
                                f"[WS] Queue: pending={data.get('pending')}, "
                                f"speaking={data.get('speaking')}"
                            )
                        elif msg_type == "skipped":
                            self._log_signal.emit(
                                f"[WS] Skipped: \"{data.get('text', '')[:40]}\" "
                                f"reason={data.get('reason')}"
                            )
                        elif msg_type == "error":
                            self._log_signal.emit(f"[WS] Error: {data.get('detail')}")
                        else:
                            self._log_signal.emit(f"[WS] {msg[:200]}")
                except TimeoutError:
                    continue
                except Exception as e:
                    if self._ws_chat_running:
                        self._log_signal.emit(f"[WS] Read error: {e}")
                    break

        except Exception as e:
            self._log_signal.emit(f"[WS] Connection error: {e}")
        finally:
            self._ws_chat_running = False
            self._ws_chat = None
            self._ws_status_signal.emit("disconnected")

    def _on_ws_chat_disconnect(self) -> None:
        self._ws_chat_running = False
        if self._ws_chat:
            try:
                self._ws_chat.close()
            except Exception:
                pass

    def _on_ws_chat_speak(self) -> None:
        text = self._ws_text.text().strip()
        if not text or not self._ws_chat:
            return
        priority = self._ws_priority.currentIndex()
        msg = {
            "type": "speak",
            "text": text,
            "emotion": self._ws_emotion.currentText(),
            "priority": priority,
        }
        try:
            self._ws_chat.send(json.dumps(msg))
            self._append_log(f"[WS] Sent speak: \"{text[:40]}\" priority={priority}")
            self._ws_text.clear()
        except Exception as e:
            self._append_log(f"[WS] Send error: {e}")

    def _on_ws_chat_cancel(self) -> None:
        if self._ws_chat:
            try:
                self._ws_chat.send(json.dumps({"type": "cancel"}))
                self._append_log("[WS] Cancel sent")
            except Exception as e:
                self._append_log(f"[WS] Cancel error: {e}")

    def _on_ws_status(self, status: str) -> None:
        self._ws_status_label.setText(f"WS: {status}")
        connected = status == "connected"
        self._btn_ws_connect.setEnabled(not connected)
        self._btn_ws_disconnect.setEnabled(connected)
        self._btn_ws_cancel.setEnabled(connected)

    def _on_ws_audio_received(self, audio_bytes: bytes) -> None:
        """Play received WS audio chunks."""
        try:
            import numpy as np
            import sounddevice as sd
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            sd.play(audio, 24000)
        except Exception:
            pass

    # ==================================================================
    # VC Stream (WS /vc/stream)
    # ==================================================================

    def _on_vc_browse_speaker(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Speaker File", "",
            "Speaker Files (*.tmrvc_speaker *.npy);;All Files (*)",
        )
        if path:
            self._vc_speaker_edit.setText(path)

    def _on_vc_connect(self) -> None:
        if self._ws_vc_running:
            return
        api_key = self._vc_api_key.text().strip()
        speaker_path = self._vc_speaker_edit.text().strip()
        if not api_key:
            self._append_log("ERROR: API key required for VC stream.")
            return
        if not speaker_path:
            self._append_log("ERROR: Speaker file required.")
            return
        threading.Thread(
            target=self._vc_stream_worker,
            args=(api_key, speaker_path),
            daemon=True,
        ).start()

    def _vc_stream_worker(self, api_key: str, speaker_path: str) -> None:
        try:
            import numpy as np
        except ImportError:
            self._log_signal.emit("ERROR: numpy required")
            return
        try:
            import websockets.sync.client as ws_client
        except ImportError:
            self._log_signal.emit("ERROR: 'websockets' package required")
            return

        # Load speaker embedding
        try:
            spk_embed = np.load(speaker_path).astype(np.float32)
            if spk_embed.shape != (192,):
                self._log_signal.emit(f"ERROR: Speaker embedding must be 192-dim, got {spk_embed.shape}")
                return
        except Exception as e:
            self._log_signal.emit(f"ERROR: Failed to load speaker file: {e}")
            return

        url = f"{self._ws_url()}/vc/stream?api_key={api_key}"
        self._log_signal.emit("[VC] Connecting...")
        try:
            self._ws_vc = ws_client.connect(url)
            self._ws_vc_running = True

            # Send speaker embedding (768 bytes = 192 * float32)
            self._ws_vc.send(spk_embed.tobytes())

            # Receive session ID
            session_id_bytes = self._ws_vc.recv(timeout=10)
            session_id = session_id_bytes.decode("utf-8").rstrip("\x00") if isinstance(session_id_bytes, bytes) else str(session_id_bytes)
            self._log_signal.emit(f"[VC] Session started: {session_id}")
            self._ws_status_signal.emit("VC: connected")

            # Real-time audio streaming would go here
            # For now, we keep the connection open for manual testing
            while self._ws_vc_running:
                try:
                    data = self._ws_vc.recv(timeout=1.0)
                    if isinstance(data, bytes):
                        self._ws_audio_signal.emit(data)
                except TimeoutError:
                    continue
                except Exception:
                    break

        except Exception as e:
            self._log_signal.emit(f"[VC] Error: {e}")
        finally:
            self._ws_vc_running = False
            self._ws_vc = None
            self._log_signal.emit("[VC] Disconnected")

    def _on_vc_disconnect(self) -> None:
        self._ws_vc_running = False
        if self._ws_vc:
            try:
                self._ws_vc.close()
            except Exception:
                pass

    def _on_vc_stats(self) -> None:
        threading.Thread(target=self._vc_stats_worker, daemon=True).start()

    def _vc_stats_worker(self) -> None:
        try:
            resp = self._http_get("/vc/stats")
            self._log_signal.emit(f"[VC stats] {json.dumps(resp, indent=2)}")
        except Exception as e:
            self._log_signal.emit(f"[VC stats] {self._format_error(e)}")

    # ==================================================================
    # Characters
    # ==================================================================

    def _on_refresh_characters(self) -> None:
        threading.Thread(target=self._refresh_chars_worker, daemon=True).start()

    def _refresh_chars_worker(self) -> None:
        try:
            chars = self._http_get("/characters")
            self._characters_signal.emit(json.dumps(chars or []))
            formatted = json.dumps(chars, indent=2, ensure_ascii=False)
            self._log_signal.emit(f"Characters refreshed: {len(chars or [])} found")
            # Update the character list view on the Characters tab via log signal
            # (we reuse _characters_signal for combo updates)
            self._log_signal.emit(f"[Characters]\n{formatted}")
        except Exception as e:
            self._log_signal.emit(f"Characters error: {self._format_error(e)}")

    def _on_register_character(self) -> None:
        cid = self._reg_id.text().strip()
        name = self._reg_name.text().strip()
        if not cid or not name:
            self._append_log("ERROR: Character ID and Name are required.")
            return
        body: dict = {
            "id": cid,
            "name": name,
            "personality": self._reg_personality.text().strip(),
            "voice_description": self._reg_voice_desc.text().strip(),
            "language": self._reg_language.currentText(),
        }
        spk = self._reg_speaker_file.text().strip()
        if spk:
            body["speaker_file"] = spk
        threading.Thread(
            target=self._register_char_worker, args=(body,), daemon=True,
        ).start()

    def _register_char_worker(self, body: dict) -> None:
        try:
            resp = self._http_post("/characters", body)
            self._log_signal.emit(f"Character registered: {json.dumps(resp, ensure_ascii=False)}")
            # Refresh list
            self._refresh_chars_worker()
        except Exception as e:
            self._log_signal.emit(f"Register error: {self._format_error(e)}")

    # ==================================================================
    # Auth
    # ==================================================================

    def _on_login(self) -> None:
        email = self._auth_email.text().strip()
        password = self._auth_password.text()
        if not email or not password:
            self._append_log("ERROR: Email and password required.")
            return
        threading.Thread(
            target=self._login_worker, args=(email, password), daemon=True,
        ).start()

    def _login_worker(self, email: str, password: str) -> None:
        try:
            resp = self._http_post("/auth/token", {"email": email, "password": password})
            self._auth_token = resp.get("access_token")
            expires = resp.get("expires_in", 0)
            self._log_signal.emit(f"Login OK: expires_in={expires}s")
        except Exception as e:
            self._log_signal.emit(f"Login error: {self._format_error(e)}")

    def _on_logout(self) -> None:
        if not self._auth_token:
            return
        threading.Thread(target=self._logout_worker, daemon=True).start()

    def _logout_worker(self) -> None:
        try:
            resp = self._http_post("/auth/logout", {})
            self._auth_token = None
            self._log_signal.emit(f"Logout: {resp.get('message', 'ok')}")
        except Exception as e:
            self._log_signal.emit(f"Logout error: {self._format_error(e)}")

    def _on_auth_me(self) -> None:
        threading.Thread(target=self._auth_get_worker, args=("/auth/me",), daemon=True).start()

    def _on_auth_usage(self) -> None:
        threading.Thread(target=self._auth_get_worker, args=("/auth/usage",), daemon=True).start()

    def _on_auth_keys(self) -> None:
        threading.Thread(target=self._auth_get_worker, args=("/auth/keys",), daemon=True).start()

    def _auth_get_worker(self, path: str) -> None:
        try:
            resp = self._http_get(path)
            self._log_signal.emit(f"[{path}]\n{json.dumps(resp, indent=2, ensure_ascii=False)}")
        except Exception as e:
            self._log_signal.emit(f"[{path}] {self._format_error(e)}")

    def _on_create_key(self) -> None:
        user_id = self._key_user_id.text().strip()
        if not user_id:
            self._append_log("ERROR: User ID required.")
            return
        body: dict = {
            "user_id": user_id,
            "role": self._key_role.currentText(),
        }
        days = self._key_expires.value()
        if days > 0:
            body["expires_days"] = days
        threading.Thread(
            target=self._create_key_worker, args=(body,), daemon=True,
        ).start()

    def _create_key_worker(self, body: dict) -> None:
        try:
            resp = self._http_post("/auth/keys", body)
            self._log_signal.emit(
                f"API Key created!\n"
                f"  key: {resp.get('api_key', '???')}\n"
                f"  prefix: {resp.get('key_prefix')}\n"
                f"  role: {resp.get('role')}\n"
                f"  expires: {resp.get('expires_at', 'never')}"
            )
        except Exception as e:
            self._log_signal.emit(f"Create key error: {self._format_error(e)}")

    def _on_revoke_key(self) -> None:
        prefix = self._revoke_prefix.text().strip()
        if not prefix:
            self._append_log("ERROR: Key prefix required.")
            return
        threading.Thread(
            target=self._revoke_key_worker, args=(prefix,), daemon=True,
        ).start()

    def _revoke_key_worker(self, prefix: str) -> None:
        try:
            resp = self._http_delete(f"/auth/keys/{prefix}")
            self._log_signal.emit(f"Key revoked: {resp}")
        except Exception as e:
            self._log_signal.emit(f"Revoke error: {self._format_error(e)}")
