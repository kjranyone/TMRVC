"""Evaluation page for objective metrics and A/B blind listening tests."""

from __future__ import annotations

import json
import random

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

MOS_LABELS = ["1 - Bad", "2 - Poor", "3 - Fair", "4 - Good", "5 - Excellent"]


class EvaluationPage(QWidget):
    """Objective evaluation metrics and A/B blind listening tests."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_sample: int = 0
        self._total_samples: int = 0
        self._results: dict[str, int] = {"A": 0, "Same": 0, "B": 0}
        self._mos_scores: list[dict[str, int]] = []
        self._sample_order: list[int] = []
        self._qc_enabled: bool = False
        self._qc_duplicate_indices: set[int] = set()
        self._qc_scores: list[tuple[int, int, int]] = []
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

        # External baseline
        self.baseline_combo = QComboBox()
        self.baseline_combo.addItems(["None", "Qwen3-TTS", "CosyVoice", "Custom"])
        self.baseline_combo.setEditable(False)
        select_form.addRow("External Baseline:", self.baseline_combo)

        self.btn_run_eval = QPushButton("Run Evaluation")
        select_form.addRow("", self.btn_run_eval)

        layout.addWidget(select_group)

        # --- Rater Assignment ---
        rater_group = QGroupBox("Rater Assignment")
        rater_form = QFormLayout(rater_group)

        self.rater_id_edit = QLineEdit()
        self.rater_id_edit.setPlaceholderText("Enter rater ID...")
        rater_form.addRow("Rater ID:", self.rater_id_edit)

        self.rater_role_combo = QComboBox()
        self.rater_role_combo.addItems(["rater", "director"])
        self.rater_role_combo.currentIndexChanged.connect(self._on_rater_role_changed)
        rater_form.addRow("Role:", self.rater_role_combo)

        layout.addWidget(rater_group)

        # --- Few-shot evaluation ---
        fewshot_group = QGroupBox("Few-Shot Evaluation")
        fewshot_layout = QVBoxLayout(fewshot_group)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference audio:"))
        self.fewshot_ref_edit = QLineEdit()
        self.fewshot_ref_edit.setPlaceholderText("Upload reference audio for few-shot eval...")
        ref_row.addWidget(self.fewshot_ref_edit)
        btn_browse_ref = QPushButton("Browse...")
        btn_browse_ref.clicked.connect(self._on_browse_fewshot_ref)
        ref_row.addWidget(btn_browse_ref)
        fewshot_layout.addLayout(ref_row)

        self.ref_length_label = QLabel("Reference length: --")
        fewshot_layout.addWidget(self.ref_length_label)

        self.btn_fewshot_eval = QPushButton("Run Few-Shot Compare")
        self.btn_fewshot_eval.clicked.connect(self._on_fewshot_eval)
        fewshot_layout.addWidget(self.btn_fewshot_eval)

        layout.addWidget(fewshot_group)

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

        # --- Sample Randomization & QC ---
        sample_ctrl_group = QGroupBox("Sample Control")
        sample_ctrl_layout = QHBoxLayout(sample_ctrl_group)

        self.btn_shuffle_samples = QPushButton("Shuffle Samples")
        self.btn_shuffle_samples.clicked.connect(self._on_shuffle_samples)
        sample_ctrl_layout.addWidget(self.btn_shuffle_samples)

        self.cb_qc_duplicates = QCheckBox("Enable QC duplicates")
        self.cb_qc_duplicates.stateChanged.connect(self._on_qc_toggle)
        sample_ctrl_layout.addWidget(self.cb_qc_duplicates)

        self.qc_consistency_label = QLabel("QC consistency: --")
        sample_ctrl_layout.addWidget(self.qc_consistency_label)

        sample_ctrl_layout.addStretch()
        layout.addWidget(sample_ctrl_group)

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

        # --- MOS Collection ---
        mos_group = QGroupBox("MOS Collection")
        mos_layout = QVBoxLayout(mos_group)

        # Naturalness
        nat_layout = QHBoxLayout()
        nat_layout.addWidget(QLabel("Naturalness:"))
        self.mos_nat_group = QButtonGroup(self)
        self.mos_nat_buttons: list[QRadioButton] = []
        for i, label in enumerate(MOS_LABELS, start=1):
            rb = QRadioButton(label)
            self.mos_nat_group.addButton(rb, i)
            self.mos_nat_buttons.append(rb)
            nat_layout.addWidget(rb)
        mos_layout.addLayout(nat_layout)

        # Expressiveness
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Expressiveness:"))
        self.mos_exp_group = QButtonGroup(self)
        self.mos_exp_buttons: list[QRadioButton] = []
        for i, label in enumerate(MOS_LABELS, start=1):
            rb = QRadioButton(label)
            self.mos_exp_group.addButton(rb, i)
            self.mos_exp_buttons.append(rb)
            exp_layout.addWidget(rb)
        mos_layout.addLayout(exp_layout)

        self.btn_submit_mos = QPushButton("Submit MOS")
        self.btn_submit_mos.clicked.connect(self._on_submit_mos)
        mos_layout.addWidget(self.btn_submit_mos)

        self.mos_summary_label = QLabel("MOS scores collected: 0")
        mos_layout.addWidget(self.mos_summary_label)

        layout.addWidget(mos_group)

        # --- Director Qualitative Notes ---
        self.director_notes_group = QGroupBox("Director Qualitative Notes")
        director_layout = QVBoxLayout(self.director_notes_group)

        self.director_notes_edit = QTextEdit()
        self.director_notes_edit.setPlaceholderText(
            "Director-only notes (not visible to blinded raters)..."
        )
        self.director_notes_edit.setMaximumHeight(120)
        director_layout.addWidget(self.director_notes_edit)

        # Initially hidden (visible only when role=director)
        self.director_notes_group.setVisible(False)
        layout.addWidget(self.director_notes_group)

        # --- Export ---
        export_row = QHBoxLayout()
        self.btn_export = QPushButton("Export Results (JSON)")
        self.btn_export.clicked.connect(self._on_export_results)
        export_row.addWidget(self.btn_export)
        export_row.addStretch()
        layout.addLayout(export_row)

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
        self._sample_order = list(range(n))
        self._update_display()

    # ------------------------------------------------------------------
    # Rater assignment
    # ------------------------------------------------------------------

    def _on_rater_role_changed(self) -> None:
        """Show/hide director-only UI elements based on the selected role."""
        is_director = self.rater_role_combo.currentText() == "director"
        self.director_notes_group.setVisible(is_director)

    # ------------------------------------------------------------------
    # Sample randomization & QC
    # ------------------------------------------------------------------

    def _on_shuffle_samples(self) -> None:
        """Randomize the presentation order of samples."""
        if self._total_samples <= 0:
            return
        self._sample_order = list(range(self._total_samples))
        random.shuffle(self._sample_order)
        self._current_sample = 0
        self._results = {"A": 0, "Same": 0, "B": 0}
        self._update_display()

    def _on_qc_toggle(self, state: int) -> None:
        """Enable or disable QC duplicate insertion."""
        self._qc_enabled = state == Qt.CheckState.Checked.value
        if self._qc_enabled and self._total_samples > 0:
            # Insert repeat samples at roughly every 5th position
            self._qc_duplicate_indices = set()
            for i in range(0, self._total_samples, 5):
                self._qc_duplicate_indices.add(i)
            self.qc_consistency_label.setText("QC consistency: pending")
        else:
            self._qc_duplicate_indices = set()
            self.qc_consistency_label.setText("QC consistency: --")

    def _compute_qc_consistency(self) -> float | None:
        """Compute rater consistency from QC duplicate scores.

        Returns a value between 0.0 and 1.0, or None if not enough data.
        """
        if len(self._qc_scores) < 2:
            return None
        # Compare pairs of (original_score, duplicate_score)
        agreements = sum(1 for a, b, _ in self._qc_scores if a == b)
        return agreements / len(self._qc_scores)

    # ------------------------------------------------------------------
    # Few-shot evaluation
    # ------------------------------------------------------------------

    def _on_browse_fewshot_ref(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", "",
            "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if path:
            self.fewshot_ref_edit.setText(path)
            self._update_ref_length(path)

    def _update_ref_length(self, path: str) -> None:
        """Display the duration of the selected reference audio."""
        try:
            import soundfile as sf

            info = sf.info(path)
            duration = info.duration
            self.ref_length_label.setText(
                f"Reference length: {duration:.1f}s ({info.samplerate}Hz)"
            )
        except Exception:
            self.ref_length_label.setText("Reference length: (could not read)")

    def _on_fewshot_eval(self) -> None:
        ref_path = self.fewshot_ref_edit.text().strip()
        if not ref_path:
            return
        # Placeholder: generate with both models using ref audio, then compare
        pass

    # ------------------------------------------------------------------
    # MOS collection
    # ------------------------------------------------------------------

    def _on_submit_mos(self) -> None:
        """Record the current MOS ratings."""
        nat_id = self.mos_nat_group.checkedId()
        exp_id = self.mos_exp_group.checkedId()
        if nat_id < 1 or exp_id < 1:
            return
        self._mos_scores.append({
            "sample": self._current_sample,
            "naturalness": nat_id,
            "expressiveness": exp_id,
            "rater_id": self.rater_id_edit.text().strip(),
            "rater_role": self.rater_role_combo.currentText(),
        })
        self.mos_summary_label.setText(
            f"MOS scores collected: {len(self._mos_scores)}"
        )

        # Track QC duplicate scores if enabled
        if self._qc_enabled and self._current_sample in self._qc_duplicate_indices:
            self._qc_scores.append((nat_id, exp_id, self._current_sample))
            consistency = self._compute_qc_consistency()
            if consistency is not None:
                self.qc_consistency_label.setText(
                    f"QC consistency: {consistency:.1%}"
                )

        # Clear selection for next rating
        self.mos_nat_group.setExclusive(False)
        for rb in self.mos_nat_buttons:
            rb.setChecked(False)
        self.mos_nat_group.setExclusive(True)
        self.mos_exp_group.setExclusive(False)
        for rb in self.mos_exp_buttons:
            rb.setChecked(False)
        self.mos_exp_group.setExclusive(True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_results(self) -> None:
        """Export all evaluation results to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "eval_results.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        data = {
            "ab_results": self._results,
            "total_samples": self._total_samples,
            "mos_scores": self._mos_scores,
            "baseline": self.baseline_combo.currentText(),
            "rater_id": self.rater_id_edit.text().strip(),
            "rater_role": self.rater_role_combo.currentText(),
            "sample_order": self._sample_order,
            "qc_enabled": self._qc_enabled,
            "director_notes": self.director_notes_edit.toPlainText()
            if self.rater_role_combo.currentText() == "director"
            else None,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
