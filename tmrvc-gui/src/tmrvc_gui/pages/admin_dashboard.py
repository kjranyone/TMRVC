"""System Admin dashboard: server health, VRAM, latency, model loading."""

from __future__ import annotations

import json

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


AUDIT_CRITICAL_ACTIONS = [
    "model_checkpoint_swap",
    "split_lock_override",
    "approval_policy_change",
    "dataset_purge",
    "speaker_profile_delete",
    "holdout_reassign",
]


class AdminDashboardPage(QWidget):
    """System Admin: health monitoring, model management, runtime config."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base_url: str = "http://127.0.0.1:8000"
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Server connection ---
        conn_group = QGroupBox("Server Connection")
        conn_layout = QHBoxLayout(conn_group)

        conn_layout.addWidget(QLabel("Host:"))
        self.host_edit = QLineEdit("127.0.0.1")
        self.host_edit.setFixedWidth(120)
        conn_layout.addWidget(self.host_edit)

        conn_layout.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(8000)
        conn_layout.addWidget(self.port_spin)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._on_refresh)
        conn_layout.addWidget(self.btn_refresh)

        conn_layout.addStretch()
        layout.addWidget(conn_group)

        # --- Health & Metrics ---
        metrics_group = QGroupBox("Server Health & Metrics")
        metrics_form = QFormLayout(metrics_group)

        self.health_label = QLabel("--")
        metrics_form.addRow("Health:", self.health_label)

        self.vram_label = QLabel("--")
        metrics_form.addRow("VRAM Usage:", self.vram_label)

        self.latency_label = QLabel("--")
        metrics_form.addRow("Latency (p50/p95/p99):", self.latency_label)

        layout.addWidget(metrics_group)

        # --- Model Loading ---
        model_group = QGroupBox("Model Loading")
        model_layout = QVBoxLayout(model_group)

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("Checkpoint:"))
        self.ckpt_edit = QLineEdit()
        self.ckpt_edit.setPlaceholderText("Path to checkpoint file...")
        ckpt_row.addWidget(self.ckpt_edit)
        btn_browse_ckpt = QPushButton("Browse...")
        btn_browse_ckpt.clicked.connect(self._on_browse_ckpt)
        ckpt_row.addWidget(btn_browse_ckpt)
        model_layout.addLayout(ckpt_row)

        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self._on_load_model)
        model_layout.addWidget(self.btn_load_model)

        layout.addWidget(model_group)

        # --- Runtime Contract ---
        contract_group = QGroupBox("Runtime Contract")
        contract_layout = QVBoxLayout(contract_group)

        self.contract_display = QTextEdit()
        self.contract_display.setReadOnly(True)
        self.contract_display.setMaximumHeight(200)
        self.contract_display.setPlaceholderText(
            "Current pointer/voice_state configuration will appear here..."
        )
        contract_layout.addWidget(self.contract_display)

        layout.addWidget(contract_group)

        # --- Approval Policy section ---
        policy_group = QGroupBox("Approval Policy")
        policy_layout = QVBoxLayout(policy_group)

        self.cb_double_approval = QCheckBox("Require double approval for tts_mainline")
        policy_layout.addWidget(self.cb_double_approval)

        override_row = QHBoxLayout()
        override_row.addWidget(QLabel("Override authority:"))
        self.override_authority_combo = QComboBox()
        self.override_authority_combo.addItems(["admin_only", "auditor_and_above"])
        override_row.addWidget(self.override_authority_combo)
        override_row.addStretch()
        policy_layout.addLayout(override_row)

        policy_layout.addWidget(QLabel("Audit-critical actions:"))
        self.audit_actions_list = QListWidget()
        self.audit_actions_list.setMaximumHeight(120)
        for action in AUDIT_CRITICAL_ACTIONS:
            self.audit_actions_list.addItem(QListWidgetItem(action))
        policy_layout.addWidget(self.audit_actions_list)

        self.btn_save_policy = QPushButton("Save Policy")
        self.btn_save_policy.clicked.connect(self._on_save_policy)
        policy_layout.addWidget(self.btn_save_policy)

        layout.addWidget(policy_group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_base_url(self) -> str:
        host = self.host_edit.text().strip() or "127.0.0.1"
        port = self.port_spin.value()
        return f"http://{host}:{port}"

    def _fetch_json(self, endpoint: str) -> dict | None:
        """Fetch JSON from the server. Returns None on error."""
        import urllib.request
        import urllib.error

        url = f"{self._get_base_url()}{endpoint}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_refresh(self) -> None:
        """Refresh all dashboard panels from the server."""
        # Health
        health = self._fetch_json("/admin/health")
        if health is not None:
            status = health.get("status", "unknown")
            self.health_label.setText(status)
        else:
            # Fall back to basic /health endpoint
            basic = self._fetch_json("/health")
            if basic is not None:
                self.health_label.setText(str(basic.get("status", "ok")))
            else:
                self.health_label.setText("unreachable")

        # VRAM
        metrics = self._fetch_json("/admin/metrics")
        if metrics is not None:
            vram = metrics.get("vram_mb", metrics.get("vram", "--"))
            self.vram_label.setText(f"{vram} MB" if vram != "--" else "--")

            lat = metrics.get("latency", {})
            p50 = lat.get("p50", "--")
            p95 = lat.get("p95", "--")
            p99 = lat.get("p99", "--")
            self.latency_label.setText(f"{p50}ms / {p95}ms / {p99}ms")
        else:
            self.vram_label.setText("--")
            self.latency_label.setText("--")

        # Runtime contract
        contract = self._fetch_json("/admin/config")
        if contract is not None:
            self.contract_display.setText(
                json.dumps(contract, indent=2, ensure_ascii=False)
            )
        else:
            self.contract_display.setText("Could not fetch runtime config.")

    def _on_browse_ckpt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", "",
            "PyTorch Checkpoint (*.pt);;All Files (*)",
        )
        if path:
            self.ckpt_edit.setText(path)

    def _on_load_model(self) -> None:
        """Request the server to load a new model checkpoint."""
        ckpt_path = self.ckpt_edit.text().strip()
        if not ckpt_path:
            return
        import urllib.request
        import urllib.error

        url = f"{self._get_base_url()}/admin/load_model"
        payload = json.dumps({"checkpoint": ckpt_path}).encode("utf-8")
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            self.contract_display.setText(
                f"Model loaded:\n{json.dumps(body, indent=2, ensure_ascii=False)}"
            )
        except Exception as e:
            self.contract_display.setText(f"Load failed: {e}")

    def _on_save_policy(self) -> None:
        """Save the current approval policy configuration."""
        policy = {
            "require_double_approval": self.cb_double_approval.isChecked(),
            "override_authority": self.override_authority_combo.currentText(),
            "audit_critical_actions": AUDIT_CRITICAL_ACTIONS,
        }
        import urllib.request
        import urllib.error

        url = f"{self._get_base_url()}/admin/policy"
        payload = json.dumps(policy).encode("utf-8")
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            self.contract_display.setText(
                f"Policy saved:\n{json.dumps(body, indent=2, ensure_ascii=False)}"
            )
        except Exception as e:
            self.contract_display.setText(f"Policy save failed: {e}")
