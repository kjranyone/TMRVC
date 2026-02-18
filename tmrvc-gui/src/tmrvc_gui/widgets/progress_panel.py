"""ProgressPanel widget - Progress bar with status text and cancel button."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ProgressPanel(QWidget):
    """A progress indicator with a textual status line and a cancel button.

    Signals:
        cancel_requested: Emitted when the user clicks *Cancel*.

    Args:
        parent: Optional parent widget.
    """

    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # --- Status label ---
        self._status_label = QLabel("Idle")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # --- Progress bar ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setEnabled(False)

        # --- Cancel button ---
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setFixedWidth(80)
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self.cancel_requested.emit)

        # --- Layout ---
        bar_layout = QHBoxLayout()
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.addWidget(self._progress_bar, stretch=1)
        bar_layout.addWidget(self._cancel_button)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._status_label)
        main_layout.addLayout(bar_layout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_progress(self, current: int, total: int) -> None:
        """Update the progress bar.

        Args:
            current: Current step (0-based or 1-based, up to *total*).
            total: Total number of steps. If zero, the bar enters
                   indeterminate (busy) mode.
        """
        if total <= 0:
            self._progress_bar.setRange(0, 0)  # indeterminate
            return

        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(min(current, total))

    def set_status(self, text: str) -> None:
        """Update the status label text.

        Args:
            text: Status message to display.
        """
        self._status_label.setText(text)

    def set_running(self, running: bool) -> None:
        """Toggle between running and idle states.

        When *running* is ``True`` the cancel button is shown and the
        progress bar is enabled.  When ``False`` the cancel button is
        hidden and the bar is disabled.

        Args:
            running: Whether an operation is currently in progress.
        """
        self._cancel_button.setVisible(running)
        self._progress_bar.setEnabled(running)

        if not running:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)
