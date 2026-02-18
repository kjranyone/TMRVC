"""Evaluation page for objective metrics and A/B blind listening tests."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class EvaluationPage(QWidget):
    """Objective evaluation metrics and A/B blind listening tests."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_sample: int = 0
        self._total_samples: int = 0
        self._results: dict[str, int] = {"A": 0, "Same": 0, "B": 0}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Model selection ---
        select_group = QGroupBox("Model Selection")
        select_form = QFormLayout(select_group)

        self.model_a_combo = QComboBox()
        self.model_a_combo.setEditable(True)
        self.model_a_combo.setPlaceholderText("Select Model A checkpoint")
        select_form.addRow("Model A:", self.model_a_combo)

        self.model_b_combo = QComboBox()
        self.model_b_combo.setEditable(True)
        self.model_b_combo.setPlaceholderText("Select Model B checkpoint")
        select_form.addRow("Model B:", self.model_b_combo)

        self.eval_set_combo = QComboBox()
        self.eval_set_combo.addItems([
            "VCTK-test",
            "JVS-test",
            "LibriTTS-R-test",
            "Custom",
        ])
        select_form.addRow("Eval set:", self.eval_set_combo)

        self.btn_run_eval = QPushButton("Run Evaluation")
        select_form.addRow("", self.btn_run_eval)

        layout.addWidget(select_group)

        # --- Objective metrics table ---
        metrics_group = QGroupBox("Objective Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_table = QTableWidget(5, 4)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Metric", "Model A", "Model B", "Target"]
        )
        header = self.metrics_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        metrics_data = [
            ("SECS", "", "", ">= 0.85"),
            ("UTMOS", "", "", ">= 4.0"),
            ("MCD (dB)", "", "", "<= 6.0"),
            ("F0 RMSE (Hz)", "", "", "<= 15.0"),
            ("Latency (ms)", "", "", "<= 50"),
        ]
        for row, (name, a_val, b_val, target) in enumerate(metrics_data):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(a_val))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(b_val))
            self.metrics_table.setItem(row, 3, QTableWidgetItem(target))

        metrics_layout.addWidget(self.metrics_table)
        layout.addWidget(metrics_group)

        # --- A/B blind listening test ---
        ab_group = QGroupBox("A/B Blind Listening Test")
        ab_layout = QVBoxLayout(ab_group)

        # Sample counter
        counter_layout = QHBoxLayout()
        self.sample_counter_label = QLabel("Sample: 0 / 0")
        counter_layout.addWidget(self.sample_counter_label)
        counter_layout.addStretch()
        ab_layout.addLayout(counter_layout)

        # Play buttons
        play_layout = QHBoxLayout()
        self.btn_play_source = QPushButton("Play Source")
        self.btn_play_a = QPushButton("Play A")
        self.btn_play_b = QPushButton("Play B")
        play_layout.addWidget(self.btn_play_source)
        play_layout.addWidget(self.btn_play_a)
        play_layout.addWidget(self.btn_play_b)
        ab_layout.addLayout(play_layout)

        # Choice buttons
        choice_layout = QHBoxLayout()
        choice_layout.addStretch()
        self.btn_choose_a = QPushButton("Prefer A")
        self.btn_choose_same = QPushButton("Same / No Preference")
        self.btn_choose_b = QPushButton("Prefer B")
        choice_layout.addWidget(self.btn_choose_a)
        choice_layout.addWidget(self.btn_choose_same)
        choice_layout.addWidget(self.btn_choose_b)
        choice_layout.addStretch()
        ab_layout.addLayout(choice_layout)

        # Connect choice buttons
        self.btn_choose_a.clicked.connect(lambda: self._record_choice("A"))
        self.btn_choose_same.clicked.connect(lambda: self._record_choice("Same"))
        self.btn_choose_b.clicked.connect(lambda: self._record_choice("B"))

        # Results summary
        self.results_label = QLabel("Results: A=0  Same=0  B=0")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ab_layout.addWidget(self.results_label)

        layout.addWidget(ab_group)

    # ------------------------------------------------------------------
    # Slots / helpers
    # ------------------------------------------------------------------

    def _record_choice(self, choice: str) -> None:
        """Record an A/B test choice and advance to the next sample."""
        self._results[choice] = self._results.get(choice, 0) + 1
        self._current_sample += 1
        self._update_display()

    def _update_display(self) -> None:
        """Refresh the sample counter and results labels."""
        self.sample_counter_label.setText(
            f"Sample: {self._current_sample} / {self._total_samples}"
        )
        self.results_label.setText(
            f"Results: A={self._results['A']}  "
            f"Same={self._results['Same']}  "
            f"B={self._results['B']}"
        )

    def set_total_samples(self, n: int) -> None:
        """Set the total number of test samples."""
        self._total_samples = n
        self._current_sample = 0
        self._results = {"A": 0, "Same": 0, "B": 0}
        self._update_display()
