"""Base worker class for all TMRVC background jobs.

Provides common signal definitions, cancellation support, and a standard
lifecycle for long-running operations executed off the GUI thread.
"""

from __future__ import annotations

import threading
import traceback
from abc import abstractmethod

from PySide6.QtCore import QThread, Signal


class BaseWorker(QThread):
    """Abstract base class for background workers.

    Subclasses must implement :meth:`run` and should periodically check
    :attr:`is_cancelled` to support cooperative cancellation.

    Signals
    -------
    progress(int, int)
        Emitted to report progress as ``(current, total)``.
    log_message(str)
        Emitted for human-readable log lines.
    metric(str, float, int)
        Emitted for scalar metrics as ``(name, value, step)``.
    finished(bool, str)
        Emitted when the worker completes.  ``(success, message)``.
    error(str)
        Emitted when an unrecoverable error occurs.
    """

    progress = Signal(int, int)
    log_message = Signal(str)
    metric = Signal(str, float, int)
    finished = Signal(bool, str)
    error = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cancelled = threading.Event()

    # ------------------------------------------------------------------
    # Cancellation helpers
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cooperative cancellation of the running task."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        """Return *True* if cancellation has been requested."""
        return self._cancelled.is_set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> None:  # pragma: no cover
        """Execute the background task.

        Subclasses **must** override this method.  Implementations should:

        * Call ``self.progress.emit(current, total)`` periodically.
        * Check ``self.is_cancelled`` and exit early when *True*.
        * Call ``self.finished.emit(True, msg)`` on success.
        * Call ``self.error.emit(msg)`` followed by
          ``self.finished.emit(False, msg)`` on failure.
        """
        ...

    def _safe_run(self, func, *args, **kwargs):
        """Run *func* inside a try/except and emit error signals on failure.

        This is a convenience wrapper that subclasses can use inside their
        :meth:`run` implementations.
        """
        try:
            func(*args, **kwargs)
        except Exception as exc:
            tb = traceback.format_exc()
            msg = f"{exc}\n{tb}"
            self.error.emit(msg)
            self.finished.emit(False, str(exc))
