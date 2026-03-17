
"""Dataset adapters for loading various raw data layouts (Minimal SOTA)."""

from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional
from tmrvc_core.types import Utterance

class DatasetAdapter:
    """Base class for dataset adapters."""
    name = "generic"

    def iter_utterances(self, root: Path, split: str = "train") -> Iterator[Utterance]:
        """Iterate over utterances in the dataset."""
        # SOTA: Basic recursive scan for any .wav/.flac files
        search_root = root / split if (root / split).is_dir() else root
        
        found_any = False
        for p in search_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".wav", ".flac"):
                found_any = True
                # Use parent name as speaker_id, or "default_speaker" if root
                speaker_id = p.parent.name if p.parent != search_root else "default_speaker"
                yield Utterance(
                    utterance_id=p.stem,
                    speaker_id=speaker_id,
                    audio_path=str(p),
                    text=None
                )
        
        if not found_any:
            # Fallback: scan root itself if rglob failed to find nested files
            for p in search_root.iterdir():
                if p.is_file() and p.suffix.lower() in (".wav", ".flac"):
                    yield Utterance(
                        utterance_id=p.stem,
                        speaker_id="default_speaker",
                        audio_path=str(p),
                        text=None
                    )

class VCTKAdapter(DatasetAdapter):
    name = "vctk"

class JVSAdapter(DatasetAdapter):
    name = "jvs"

class TsukuyomiAdapter(DatasetAdapter):
    name = "tsukuyomi"

def get_adapter(name: str = "generic", adapter_type: str | None = None, **kwargs) -> DatasetAdapter:
    """Factory to get the appropriate adapter by name."""
    # SOTA: Prioritize name, then adapter_type
    key = name if name != "generic" else (adapter_type or "generic")
    
    mapping = {
        "vctk": VCTKAdapter,
        "jvs": JVSAdapter,
        "tsukuyomi": TsukuyomiAdapter,
        "generic": DatasetAdapter
    }
    return mapping.get(key, DatasetAdapter)()
