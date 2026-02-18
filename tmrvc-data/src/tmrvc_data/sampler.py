"""Balanced speaker sampler for training DataLoader."""

from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler


class BalancedSpeakerSampler(Sampler[int]):
    """Sample indices so that each speaker is represented roughly equally.

    Strategy: round-robin over speakers, picking one utterance from each
    speaker in turn.  Each index is emitted exactly once per epoch.
    Speakers with fewer utterances are exhausted first and removed from
    the rotation; the remaining speakers continue until all indices are
    emitted.
    """

    def __init__(self, speaker_ids: list[str], seed: int | None = None) -> None:
        self._speaker_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, sid in enumerate(speaker_ids):
            self._speaker_to_indices[sid].append(idx)
        self._speakers = sorted(self._speaker_to_indices.keys())
        self._total = len(speaker_ids)
        self._seed = seed

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
                if ptr < len(pool):
                    yield pool[ptr]
                    ptrs[sid] = ptr + 1
                    if ptr + 1 < len(pool):
                        next_active.append(sid)
            active = next_active
