"""FileDropArea widget - Drag-and-drop zone for audio files."""

from __future__ import annotations

import os
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


_DEFAULT_EXTENSIONS = [".wav", ".flac", ".mp3"]

_STYLE_NORMAL = """
    FileDropArea {
        border: 2px dashed #666;
        border-radius: 8px;
        background-color: #2a2a2a;
    }
"""

_STYLE_HOVER = """
    FileDropArea {
        border: 2px dashed #4fc3f7;
        border-radius: 8px;
        background-color: #333;
    }
"""


class FileDropArea(QFrame):
    """Drag-and-drop area that accepts audio files.

    When files with matching extensions are dropped, the
    :pyqt:`files_dropped` signal is emitted with the list of absolute
    file-path strings.

    Args:
        accepted_extensions: Lowercase file extensions to accept,
            including the leading dot (e.g. ``[".wav", ".flac"]``).
        prompt_text: Instructional text displayed inside the drop area.
        parent: Optional parent widget.
    """

    files_dropped = Signal(list)

    def __init__(
        self,
        accepted_extensions: List[str] | None = None,
        prompt_text: str = "Drop audio files here",
        parent=None,
    ) -> None:
        super().__init__(parent)

        self._accepted: List[str] = [
            ext.lower() for ext in (accepted_extensions or _DEFAULT_EXTENSIONS)
        ]

        self.setAcceptDrops(True)
        self.setMinimumHeight(80)
        self.setStyleSheet(_STYLE_NORMAL)

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel(prompt_text)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

    # ------------------------------------------------------------------
    # Drag / Drop events
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        """Accept the drag if it carries URLs with valid extensions."""
        if event.mimeData().hasUrls():
            # Accept if at least one URL has a matching extension.
            for url in event.mimeData().urls():
                if url.isLocalFile() and self._has_valid_ext(url.toLocalFile()):
                    event.acceptProposedAction()
                    self.setStyleSheet(_STYLE_HOVER)
                    return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        """Restore normal styling when the drag leaves the widget."""
        self.setStyleSheet(_STYLE_NORMAL)
        event.accept()

    def dropEvent(self, event) -> None:  # noqa: N802
        """Filter dropped files by extension and emit *files_dropped*."""
        self.setStyleSheet(_STYLE_NORMAL)

        if not event.mimeData().hasUrls():
            event.ignore()
            return

        accepted_paths: List[str] = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if self._has_valid_ext(path):
                    accepted_paths.append(os.path.normpath(path))

        if accepted_paths:
            event.acceptProposedAction()
            self.files_dropped.emit(accepted_paths)
        else:
            event.ignore()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_valid_ext(self, path: str) -> bool:
        """Return True if *path* ends with an accepted extension."""
        _, ext = os.path.splitext(path)
        return ext.lower() in self._accepted
