"""AudioMeter widget - Audio level meter with colour-coded bar."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

# dB range rendered by the meter.
_DB_MIN = -60.0
_DB_MAX = 0.0


def _db_to_percent(db: float) -> int:
    """Map a dB value in [_DB_MIN, _DB_MAX] to an integer in [0, 100]."""
    clamped = max(_DB_MIN, min(_DB_MAX, db))
    return int((clamped - _DB_MIN) / (_DB_MAX - _DB_MIN) * 100)


def _color_for_db(db: float) -> str:
    """Return a CSS colour string for the given dB level.

    * Green  for levels below -20 dB
    * Yellow for levels between -20 dB and -6 dB
    * Red    for levels above -6 dB
    """
    if db > -6:
        return "#e74c3c"  # red
    elif db > -20:
        return "#f1c40f"  # yellow
    else:
        return "#2ecc71"  # green


class AudioMeter(QWidget):
    """Vertical or horizontal audio-level meter.

    The meter maps a dB value (from -60 to 0) onto a coloured progress bar
    and displays the numeric dB reading as a text label.

    Args:
        orientation: ``"horizontal"`` or ``"vertical"``.
        label: Optional text label shown next to the meter.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        orientation: str = "horizontal",
        label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._db: float = _DB_MIN

        # --- Progress bar ---
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)

        if orientation == "vertical":
            self._bar.setOrientation(Qt.Orientation.Vertical)
            self._bar.setMinimumHeight(120)
            self._bar.setMaximumWidth(24)
        else:
            self._bar.setOrientation(Qt.Orientation.Horizontal)
            self._bar.setMinimumWidth(120)
            self._bar.setMaximumHeight(24)

        # --- dB readout ---
        self._db_label = QLabel(f"{_DB_MIN:.1f} dB")
        self._db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._db_label.setMinimumWidth(60)

        # --- Optional name label ---
        self._name_label: QLabel | None = None
        if label:
            self._name_label = QLabel(label)
            self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Layout ---
        if orientation == "vertical":
            layout = QVBoxLayout(self)
            layout.setContentsMargins(2, 2, 2, 2)
            if self._name_label is not None:
                layout.addWidget(self._name_label)
            layout.addWidget(self._bar, stretch=1)
            layout.addWidget(self._db_label)
        else:
            layout = QHBoxLayout(self)
            layout.setContentsMargins(2, 2, 2, 2)
            if self._name_label is not None:
                layout.addWidget(self._name_label)
            layout.addWidget(self._bar, stretch=1)
            layout.addWidget(self._db_label)

        # Apply initial colour.
        self._apply_colour()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_level(self, db: float) -> None:
        """Set the displayed audio level.

        Args:
            db: Level in decibels.  Clamped to the range [-60, 0].
        """
        self._db = max(_DB_MIN, min(_DB_MAX, db))
        self._bar.setValue(_db_to_percent(self._db))
        self._db_label.setText(f"{self._db:.1f} dB")
        self._apply_colour()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_colour(self) -> None:
        """Update the progress bar colour based on the current dB level."""
        colour = _color_for_db(self._db)
        self._bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #222;
            }}
            QProgressBar::chunk {{
                background-color: {colour};
            }}
            """
        )
