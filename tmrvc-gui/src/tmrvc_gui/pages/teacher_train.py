"""Teacher training page with phase selection, hyperparameters, and metric plots."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _make_plot_placeholder(title: str) -> QFrame:
    """Create a placeholder QFrame for a future pyqtgraph plot."""
    frame = QFrame()
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setMinimumSize(200, 120)
    frame_layout = QVBoxLayout(frame)
    label = QLabel(f"Plot: {title}")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    frame_layout.addWidget(label)
    return frame


class TeacherTrainPage(QWidget):
    """Teacher (diffusion U-Net) training configuration and monitoring."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        top_row = QHBoxLayout()

        # --- Phase / dataset / hyperparameters ---
        config_group = QGroupBox("Training Configuration")
        config_form = QFormLayout(config_group)

        # Phase selection
        self.phase_combo = QComboBox()
        self.phase_combo.addItems([
            "Phase 0 — Warm-up (VCTK + JVS)",
            "Phase 1 — Core (VCTK + JVS)",
            "Phase 2 — Expansion (+ LibriTTS-R)",
            "Phase 3 — Scale (+ Emilia)",
        ])
        config_form.addRow("Phase:", self.phase_combo)

        # Dataset checkboxes
        dataset_layout = QHBoxLayout()
        self.cb_vctk = QCheckBox("VCTK")
        self.cb_vctk.setChecked(True)
        self.cb_jvs = QCheckBox("JVS")
        self.cb_jvs.setChecked(True)
        self.cb_libritts = QCheckBox("LibriTTS-R")
        self.cb_emilia = QCheckBox("Emilia")
        dataset_layout.addWidget(self.cb_vctk)
        dataset_layout.addWidget(self.cb_jvs)
        dataset_layout.addWidget(self.cb_libritts)
        dataset_layout.addWidget(self.cb_emilia)
        config_form.addRow("Datasets:", dataset_layout)

        # Hyperparameters
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(32)
        config_form.addRow("Batch size:", self.batch_size_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-7, 1e-1)
        self.lr_spin.setDecimals(7)
        self.lr_spin.setValue(1e-4)
        self.lr_spin.setSingleStep(1e-5)
        config_form.addRow("Learning rate:", self.lr_spin)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1000, 10_000_000)
        self.steps_spin.setSingleStep(10000)
        self.steps_spin.setValue(200_000)
        config_form.addRow("Total steps:", self.steps_spin)

        # Directories
        self.cache_dir_edit = QLineEdit()
        self.cache_dir_edit.setPlaceholderText("data/cache")
        self.cache_dir_edit.setText("data/cache")
        config_form.addRow("Cache dir:", self.cache_dir_edit)

        self.ckpt_dir_edit = QLineEdit()
        self.ckpt_dir_edit.setPlaceholderText("checkpoints")
        self.ckpt_dir_edit.setText("checkpoints")
        config_form.addRow("Checkpoint dir:", self.ckpt_dir_edit)

        self.resume_edit = QLineEdit()
        self.resume_edit.setPlaceholderText("(optional) path to resume checkpoint")
        config_form.addRow("Resume from:", self.resume_edit)

        top_row.addWidget(config_group)

        # --- Execution mode ---
        exec_group = QGroupBox("Execution")
        exec_layout = QVBoxLayout(exec_group)

        self.exec_button_group = QButtonGroup(self)
        self.radio_local = QRadioButton("Local")
        self.radio_local.setChecked(True)
        self.radio_ssh = QRadioButton("SSH Remote")
        self.exec_button_group.addButton(self.radio_local)
        self.exec_button_group.addButton(self.radio_ssh)
        exec_layout.addWidget(self.radio_local)
        exec_layout.addWidget(self.radio_ssh)

        # SSH config fields
        ssh_form = QFormLayout()
        self.ssh_host_edit = QLineEdit()
        self.ssh_host_edit.setPlaceholderText("hostname or IP")
        self.ssh_host_edit.setEnabled(False)
        ssh_form.addRow("Host:", self.ssh_host_edit)

        self.ssh_user_edit = QLineEdit()
        self.ssh_user_edit.setPlaceholderText("username")
        self.ssh_user_edit.setEnabled(False)
        ssh_form.addRow("User:", self.ssh_user_edit)

        self.ssh_key_edit = QLineEdit()
        self.ssh_key_edit.setPlaceholderText("~/.ssh/id_rsa")
        self.ssh_key_edit.setEnabled(False)
        ssh_form.addRow("Key:", self.ssh_key_edit)

        self.ssh_remote_dir_edit = QLineEdit()
        self.ssh_remote_dir_edit.setPlaceholderText("/home/user/TMRVC")
        self.ssh_remote_dir_edit.setEnabled(False)
        ssh_form.addRow("Remote dir:", self.ssh_remote_dir_edit)

        exec_layout.addLayout(ssh_form)

        # Connect radio toggle to enable/disable SSH fields
        self.radio_ssh.toggled.connect(self._on_ssh_toggled)

        # Start / pause / stop buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        exec_layout.addLayout(btn_layout)

        top_row.addWidget(exec_group)
        layout.addLayout(top_row)

        # --- Metric plots (2x2 grid) ---
        plots_group = QGroupBox("Training Metrics")
        plots_layout = QHBoxLayout(plots_group)

        left_col = QVBoxLayout()
        self.plot_loss = _make_plot_placeholder("Loss")
        left_col.addWidget(self.plot_loss)
        self.plot_secs = _make_plot_placeholder("SECS (Speaker Embedding Cosine Similarity)")
        left_col.addWidget(self.plot_secs)

        right_col = QVBoxLayout()
        self.plot_utmos = _make_plot_placeholder("UTMOS")
        right_col.addWidget(self.plot_utmos)
        self.plot_lr = _make_plot_placeholder("LR Schedule")
        right_col.addWidget(self.plot_lr)

        plots_layout.addLayout(left_col)
        plots_layout.addLayout(right_col)
        layout.addWidget(plots_group)

        # --- Log viewer ---
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMinimumHeight(100)
        self.log_viewer.setPlaceholderText("Training logs will appear here...")
        log_layout.addWidget(self.log_viewer)
        layout.addWidget(log_group)

    # ------------------------------------------------------------------
    # Worker state
    # ------------------------------------------------------------------

    _worker = None  # type: ignore[assignment]

    # Phase combo index → phase string
    _PHASE_MAP = {0: "0", 1: "1a", 2: "1b", 3: "2"}

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_ssh_toggled(self, checked: bool) -> None:
        """Enable or disable SSH configuration fields."""
        self.ssh_host_edit.setEnabled(checked)
        self.ssh_user_edit.setEnabled(checked)
        self.ssh_key_edit.setEnabled(checked)
        self.ssh_remote_dir_edit.setEnabled(checked)

    def _on_start(self) -> None:
        from tmrvc_gui.workers.train_worker import TrainWorker

        # Gather datasets
        datasets: list[str] = []
        if self.cb_vctk.isChecked():
            datasets.append("vctk")
        if self.cb_jvs.isChecked():
            datasets.append("jvs")
        if self.cb_libritts.isChecked():
            datasets.append("libritts_r")
        if self.cb_emilia.isChecked():
            datasets.append("emilia")

        phase = self._PHASE_MAP.get(self.phase_combo.currentIndex(), "0")
        mode = "ssh" if self.radio_ssh.isChecked() else "local"

        ssh_config = None
        if mode == "ssh":
            ssh_config = {
                "host": self.ssh_host_edit.text(),
                "user": self.ssh_user_edit.text(),
                "key_path": self.ssh_key_edit.text(),
                "remote_dir": self.ssh_remote_dir_edit.text(),
            }

        resume = self.resume_edit.text().strip() or None

        config = {
            "phase": phase,
            "datasets": datasets,
            "batch_size": self.batch_size_spin.value(),
            "lr": self.lr_spin.value(),
            "total_steps": self.steps_spin.value(),
            "checkpoint_dir": self.ckpt_dir_edit.text() or "checkpoints",
            "cache_dir": self.cache_dir_edit.text() or "data/cache",
            "resume_from": resume,
            "mode": mode,
            "ssh_config": ssh_config,
        }

        self._worker = TrainWorker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self.append_log)
        self._worker.metric.connect(self._on_metric)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(lambda msg: self.append_log(f"ERROR: {msg}"))

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _on_progress(self, current: int, total: int) -> None:
        self.append_log(f"Step {current}/{total}")

    def _on_metric(self, name: str, value: float, step: int) -> None:
        # Placeholder: metrics will be routed to plots in the future
        pass

    def _on_finished(self, success: bool, message: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.append_log(f"Finished: {message}")
        self._worker = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def append_log(self, text: str) -> None:
        """Append a line to the log viewer."""
        self.log_viewer.append(text)
