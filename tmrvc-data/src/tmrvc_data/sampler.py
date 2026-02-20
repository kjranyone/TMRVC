"""Balanced speaker sampler for training DataLoader."""

from __future__ import annotations

import fnmatch
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


@dataclass
class SpeakerGroupConfig:
    """Configuration for a speaker group with sampling weight.

    Attributes:
        speakers: fnmatch patterns to match speaker IDs
                  (e.g. ``["moe/*"]`` or ``["tsukuyomi", "aoi_*"]``).
        weight: Number of utterances to yield per round-robin cycle
                (default 1).  Higher values increase the sampling
                frequency of this group without duplicating utterances.
    """

    speakers: list[str] = field(default_factory=list)
    weight: int = 1


class BalancedSpeakerSampler(Sampler[int]):
    """Sample indices so that each speaker is represented roughly equally.

    Strategy: round-robin over speakers, picking one utterance from each
    speaker in turn.  Each index is emitted exactly once per epoch.
    Speakers with fewer utterances are exhausted first and removed from
    the rotation; the remaining speakers continue until all indices are
    emitted.

    When *speaker_groups* is provided, matched speakers yield *weight*
    utterances per round instead of 1, effectively increasing their
    representation without duplicating any utterance.
    """

    def __init__(
        self,
        speaker_ids: list[str],
        seed: int | None = None,
        speaker_groups: list[SpeakerGroupConfig] | None = None,
    ) -> None:
        self._speaker_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, sid in enumerate(speaker_ids):
            self._speaker_to_indices[sid].append(idx)
        self._speakers = sorted(self._speaker_to_indices.keys())
        self._total = len(speaker_ids)
        self._seed = seed

        # Build speaker → weight map from groups (fnmatch patterns)
        self._weights: dict[str, int] = {}
        for group in speaker_groups or []:
            matched_sids: list[str] = []
            for pattern in group.speakers:
                hits = fnmatch.filter(self._speaker_to_indices.keys(), pattern)
                for sid in hits:
                    self._weights[sid] = group.weight
                matched_sids.extend(hits)
                if not hits:
                    logger.debug(
                        "Speaker group pattern '%s' matched no speakers", pattern
                    )
            if matched_sids:
                logger.info(
                    "Speaker group: matched %d speaker(s) %s, weight=%d",
                    len(matched_sids),
                    matched_sids,
                    group.weight,
                )

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        rng = random.Random(self._seed)

        # Shuffle utterances within each speaker
        pools: dict[str, list[int]] = {}
        for sid in self._speakers:
            indices = list(self._speaker_to_indices[sid])
            rng.shuffle(indices)
            pools[sid] = indices

        # Round-robin: cycle through speakers, removing exhausted ones
        speaker_order = list(self._speakers)
        rng.shuffle(speaker_order)

        ptrs: dict[str, int] = {sid: 0 for sid in self._speakers}
        active = list(speaker_order)

        while active:
            next_active = []
            for sid in active:
                ptr = ptrs[sid]
                pool = pools[sid]
                weight = self._weights.get(sid, 1)
                # Yield up to `weight` utterances per round
                emitted = 0
                while emitted < weight and ptr < len(pool):
                    yield pool[ptr]
                    ptr += 1
                    emitted += 1
                ptrs[sid] = ptr
                if ptr < len(pool):
                    next_active.append(sid)
            active = next_active
