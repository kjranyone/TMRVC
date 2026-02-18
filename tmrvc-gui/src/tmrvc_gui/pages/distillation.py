"""Distillation page for teacher-to-student knowledge distillation."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
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


class DistillationPage(QWidget):
    """Teacher-to-student distillation configuration and A/B listening."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        top_row = QHBoxLayout()

        # --- Configuration ---
        config_group = QGroupBox("Distillation Configuration")
        config_form = QFormLayout(config_group)

        # Teacher checkpoint
        self.teacher_ckpt_combo = QComboBox()
        self.teacher_ckpt_combo.setEditable(True)
        self.teacher_ckpt_combo.setPlaceholderText("Select or type teacher checkpoint path")
        config_form.addRow("Teacher checkpoint:", self.teacher_ckpt_combo)

        # Distillation phase
        phase_layout = QHBoxLayout()
        self.phase_button_group = QButtonGroup(self)
        self.radio_phase_a = QRadioButton("A: ODE Trajectory")
        self.radio_phase_a.setChecked(True)
        self.radio_phase_b = QRadioButton("B: DMD")
        self.phase_button_group.addButton(self.radio_phase_a)
        self.phase_button_group.addButton(self.radio_phase_b)
        phase_layout.addWidget(self.radio_phase_a)
        phase_layout.addWidget(self.radio_phase_b)
        config_form.addRow("Phase:", phase_layout)

        # Student architecture
        self.student_arch_combo = QComboBox()
        self.student_arch_combo.addItems([
            "CausalCNN-7.7M (default)",
            "CausalCNN-4M (lite)",
            "CausalCNN-12M (large)",
        ])
        config_form.addRow("Student arch:", self.student_arch_combo)

        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(32)
        config_form.addRow("Batch size:", self.batch_size_spin)

        # Steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1000, 10_000_000)
        self.steps_spin.setSingleStep(10000)
        self.steps_spin.setValue(100_000)
        config_form.addRow("Total steps:", self.steps_spin)

        # Start button
        self.btn_start = QPushButton("Start Distillation")
        config_form.addRow("", self.btn_start)

        top_row.addWidget(config_group)

        # --- Metric plots ---
        plots_group = QGroupBox("Distillation Metrics")
        plots_layout = QVBoxLayout(plots_group)

        self.plot_distill_loss = _make_plot_placeholder("Distillation Loss")
        plots_layout.addWidget(self.plot_distill_loss)

        self.plot_feature_mse = _make_plot_placeholder("Feature MSE")
        plots_layout.addWidget(self.plot_feature_mse)

        top_row.addWidget(plots_group)
        layout.addLayout(top_row)

        # --- A/B listening section ---
        listen_group = QGroupBox("A/B Listening Comparison")
        listen_layout = QVBoxLayout(listen_group)

        listen_info = QLabel(
            "Compare source, teacher output, and student output for quality assessment."
        )
        listen_layout.addWidget(listen_info)

        btn_row = QHBoxLayout()

        self.btn_play_source = QPushButton("Play Source")
        btn_row.addWidget(self.btn_play_source)

        self.btn_play_teacher = QPushButton("Play Teacher")
        btn_row.addWidget(self.btn_play_teacher)

        self.btn_play_student = QPushButton("Play Student")
        btn_row.addWidget(self.btn_play_student)

        listen_layout.addLayout(btn_row)

        # Waveform display placeholders
        wave_row = QHBoxLayout()
        for label_text in ("Source Waveform", "Teacher Waveform", "Student Waveform"):
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame.setMinimumHeight(60)
            fl = QVBoxLayout(frame)
            lbl = QLabel(label_text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fl.addWidget(lbl)
            wave_row.addWidget(frame)

        listen_layout.addLayout(wave_row)
        layout.addWidget(listen_group)
