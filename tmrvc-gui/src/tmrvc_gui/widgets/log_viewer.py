"""LogViewer widget - Read-only log viewer with auto-scroll."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPlainTextEdit


class LogViewer(QPlainTextEdit):
    """Read-only log viewer with auto-scroll and line limit.

    Uses QPlainTextEdit for efficient handling of large log output.
    Automatically scrolls to the bottom when new content is appended.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._max_lines: int = 5000

        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setUndoRedoEnabled(False)

    def append_log(self, text: str) -> None:
        """Append a timestamped log line and auto-scroll to the bottom.

        Args:
            text: The log message to append. A timestamp prefix is added
                  automatically in ``HH:MM:SS.mmm`` format.
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] {text}"
        self.appendPlainText(line)
        self._enforce_max_lines()
        self._scroll_to_bottom()

    def set_max_lines(self, n: int) -> None:
        """Set the maximum number of retained lines.

        Args:
            n: Maximum line count. Must be positive.
        """
        if n < 1:
            raise ValueError("max_lines must be >= 1")
        self._max_lines = n
        self._enforce_max_lines()

    def clear_log(self) -> None:
        """Clear all log content."""
        self.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_max_lines(self) -> None:
        """Remove oldest lines when the document exceeds *max_lines*."""
        doc = self.document()
        while doc.blockCount() > self._max_lines:
            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            # Also select the trailing newline so the block is fully removed.
            cursor.movePosition(
                cursor.MoveOperation.NextCharacter,
                cursor.MoveMode.KeepAnchor,
            )
            cursor.removeSelectedText()

    def _scroll_to_bottom(self) -> None:
        """Scroll the viewport to the very bottom."""
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
