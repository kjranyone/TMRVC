"""Speaker Profile persistence and management (Worker 04/12)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tmrvc_core.types import SpeakerProfile

logger = logging.getLogger(__name__)


class CastingGalleryStore:
    """Handles persistence for SpeakerProfile objects in models/characters/."""

    def __init__(self, root_dir: str | Path = "models/characters"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, SpeakerProfile] = {}
        self.load_all()

    def _get_path(self, profile_id: str) -> Path:
        return self.root_dir / f"{profile_id}.json"

    def load_all(self) -> None:
        """Scan root_dir for all *.json speaker profiles."""
        self._profiles = {}
        for path in self.root_dir.glob("*.json"):
            try:
                profile = self.load_profile(path.stem)
                if profile:
                    self._profiles[profile.speaker_profile_id] = profile
            except Exception as e:
                logger.warning("Failed to load profile %s: %s", path, e)

    def load_profile(self, profile_id: str) -> Optional[SpeakerProfile]:
        """Load a single SpeakerProfile from disk."""
        path = self._get_path(profile_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct tensors
        if "speaker_embed" in data and data["speaker_embed"] is not None:
            data["speaker_embed"] = torch.tensor(data["speaker_embed"])
        if "prompt_codec_tokens" in data and data["prompt_codec_tokens"] is not None:
            data["prompt_codec_tokens"] = torch.tensor(data["prompt_codec_tokens"])
        return SpeakerProfile(**data)

    def save_profile(self, profile: SpeakerProfile) -> None:
        """Save a SpeakerProfile to disk as JSON."""
        path = self._get_path(profile.speaker_profile_id)
        data = asdict(profile)

        # Convert tensors to lists for JSON
        if isinstance(data["speaker_embed"], torch.Tensor):
            data["speaker_embed"] = data["speaker_embed"].tolist()
        if isinstance(data["prompt_codec_tokens"], torch.Tensor):
            data["prompt_codec_tokens"] = data["prompt_codec_tokens"].tolist()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self._profiles[profile.speaker_profile_id] = profile

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile from disk."""
        path = self._get_path(profile_id)
        if path.exists():
            path.unlink()
            self._profiles.pop(profile_id, None)
            return True
        return False

    def list_profiles(self) -> List[SpeakerProfile]:
        """Return all loaded profiles."""
        return list(self._profiles.values())

    def get_profile(self, profile_id: str) -> Optional[SpeakerProfile]:
        """Get a profile by ID from memory."""
        return self._profiles.get(profile_id)
