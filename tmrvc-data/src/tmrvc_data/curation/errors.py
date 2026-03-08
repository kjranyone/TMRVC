"""Domain errors for the curation system (Worker 07)."""

from __future__ import annotations


class CurationError(Exception):
    """Base error for the curation subsystem."""


class StaleVersionError(CurationError):
    """Raised when an update targets a record whose metadata_version has changed.

    Attributes:
        record_id: The record that was being updated.
        expected_version: The version the caller expected.
        actual_version: The version currently stored.
    """

    def __init__(
        self,
        record_id: str,
        expected_version: int,
        actual_version: int | None = None,
    ) -> None:
        self.record_id = record_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        detail = (
            f"Stale version for record '{record_id}': "
            f"expected {expected_version}"
        )
        if actual_version is not None:
            detail += f", actual {actual_version}"
        super().__init__(detail)


class InvalidTransitionError(CurationError):
    """Raised when a record status transition violates lifecycle rules."""

    def __init__(
        self,
        record_id: str,
        from_status: str,
        to_status: str,
    ) -> None:
        self.record_id = record_id
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Invalid transition for '{record_id}': "
            f"{from_status} -> {to_status}"
        )


class StageExecutionError(CurationError):
    """Raised when a stage fails to process a record."""

    def __init__(
        self,
        stage: int | str,
        record_id: str,
        cause: str,
        *,
        retryable: bool = False,
    ) -> None:
        self.stage = stage
        self.record_id = record_id
        self.cause = cause
        self.retryable = retryable
        super().__init__(
            f"Stage {stage} failed for '{record_id}': {cause}"
        )
