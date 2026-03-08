"""Stage execution framework for the AI curation pipeline (Worker 07).

Defines the base ``CurationStage`` class and a numeric stage registry
that maps stage numbers (0-9) to their implementations.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Type

from .models import CurationRecord, StageResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage name constants (canonical ordering)
# ---------------------------------------------------------------------------

STAGE_NAMES: Dict[int, str] = {
    0: "ingest",
    1: "cleanup",
    2: "separation",
    3: "speaker_recovery",
    4: "transcript_recovery",
    5: "transcript_refinement",
    6: "prosody_recovery",
    7: "quality_scoring",
    8: "promotion",
    9: "export",
}


class CurationStage(abc.ABC):
    """Base class for all curation stages.

    Subclasses must implement ``process`` which takes a single
    ``CurationRecord`` and returns a ``StageResult``.
    """

    #: Numeric stage identifier (0-9).
    stage_num: int = -1
    #: Human-readable stage name.
    stage_name: str = "unknown"

    @abc.abstractmethod
    def process(self, record: CurationRecord, **kwargs: Any) -> StageResult:
        """Execute this stage on a single record.

        Returns a ``StageResult`` indicating success/failure, outputs,
        warnings, and confidence scores.
        """
        ...

    def can_process(self, record: CurationRecord) -> bool:
        """Return True if this stage should run for the given record.

        The default implementation returns True for all records.
        Subclasses may override to skip records in incompatible states.
        """
        return True


class StageRegistry:
    """Maps stage numbers (0-9) to ``CurationStage`` implementations.

    Multiple implementations may be registered for the same stage number;
    the first registered implementation is treated as the primary.
    """

    def __init__(self) -> None:
        self._stages: Dict[int, List[CurationStage]] = {}

    def register(self, stage: CurationStage) -> None:
        """Register a stage implementation."""
        num = stage.stage_num
        if num not in self._stages:
            self._stages[num] = []
        self._stages[num].append(stage)
        logger.debug(
            "Registered stage %d (%s): %s",
            num, stage.stage_name, type(stage).__name__,
        )

    def get(self, stage_num: int) -> Optional[CurationStage]:
        """Return the primary (first-registered) implementation for *stage_num*."""
        impls = self._stages.get(stage_num)
        return impls[0] if impls else None

    def get_all(self, stage_num: int) -> List[CurationStage]:
        """Return all registered implementations for *stage_num*."""
        return list(self._stages.get(stage_num, []))

    def registered_stages(self) -> List[int]:
        """Return sorted list of stage numbers with at least one implementation."""
        return sorted(self._stages.keys())

    def __contains__(self, stage_num: int) -> bool:
        return stage_num in self._stages and len(self._stages[stage_num]) > 0


def create_default_stage_registry() -> StageRegistry:
    """Create a ``StageRegistry`` populated with built-in stub stages.

    Concrete provider-backed stages (Workers 08-10) are expected to
    register their own implementations at runtime.
    """
    registry = StageRegistry()
    # Placeholder stubs are intentionally *not* registered here so that
    # the orchestrator can detect when a stage has no implementation and
    # report it clearly rather than silently succeeding with a no-op.
    return registry
