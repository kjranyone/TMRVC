"""Shared state management for the Gradio control plane.

Handles role-based access, audit trail persistence, evaluation session
storage, and casting gallery management.
"""

from __future__ import annotations

import json
import time
import uuid
import torch
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

Role = Literal["annotator", "auditor", "director", "rater", "admin"]

ROLES: list[Role] = ["annotator", "auditor", "director", "rater", "admin"]

# Which roles can perform which actions
ROLE_PERMISSIONS: dict[str, set[Role]] = {
    "promote": {"auditor", "admin"},
    "reject": {"auditor", "admin"},
    "edit_transcript": {"annotator", "auditor", "admin"},
    "edit_speaker": {"annotator", "auditor", "admin"},
    "rate": {"rater", "director", "admin"},
    "export": {"admin"},
    "load_model": {"admin"},
    "lock_split": {"admin"},
    "override_promotion": {"admin"},
}


def check_permission(role: Role, action: str) -> bool:
    allowed = ROLE_PERMISSIONS.get(action)
    if allowed is None:
        return False
    return role in allowed


# ---------------------------------------------------------------------------
# Audit Trail
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    actor_role: str
    actor_id: str
    timestamp: float
    action: str
    before_state: str = ""
    after_state: str = ""
    rationale: str = ""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class AuditTrail:
    """Append-only audit log persisted as JSONL."""

    def __init__(self, path: Path | str = "data/audit_trail.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        actor_role: str,
        actor_id: str,
        action: str,
        before_state: str = "",
        after_state: str = "",
        rationale: str = "",
    ) -> AuditEntry:
        entry = AuditEntry(
            actor_role=actor_role,
            actor_id=actor_id,
            timestamp=time.time(),
            action=action,
            before_state=before_state,
            after_state=after_state,
            rationale=rationale,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry

    def read_recent(self, n: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        entries = [json.loads(line) for line in lines[-n:]]
        entries.reverse()
        return entries


from tmrvc_core.types import SpeakerProfile, PointerState

# ... ROLES and check_permission remain unchanged ...

# ---------------------------------------------------------------------------
# Casting Gallery (Speaker Profiles)
# ---------------------------------------------------------------------------

class CastingGallery:
    """Persistent speaker profile store."""

    def __init__(self, path: Path | str = "data/casting_gallery.json"):
        self.path = Path(path)
        self.profiles: dict[str, SpeakerProfile] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            text = self.path.read_text(encoding="utf-8").strip()
            if not text:
                return
            data = json.loads(text)
            for pid, d in data.items():
                # Reconstruct torch tensors from lists
                if "speaker_embed" in d and d["speaker_embed"] is not None:
                    d["speaker_embed"] = torch.tensor(d["speaker_embed"])
                if "prompt_codec_tokens" in d and d["prompt_codec_tokens"] is not None:
                    d["prompt_codec_tokens"] = torch.tensor(d["prompt_codec_tokens"])
                if "prompt_text_tokens" in d and d["prompt_text_tokens"] is not None:
                    d["prompt_text_tokens"] = torch.tensor(d["prompt_text_tokens"])
                
                self.profiles[pid] = SpeakerProfile(**d)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Convert tensors to lists for JSON serialization
        data = {}
        for pid, p in self.profiles.items():
            d = asdict(p)
            if isinstance(d["speaker_embed"], torch.Tensor):
                d["speaker_embed"] = d["speaker_embed"].tolist()
            if isinstance(d["prompt_codec_tokens"], torch.Tensor):
                d["prompt_codec_tokens"] = d["prompt_codec_tokens"].tolist()
            if isinstance(d["prompt_text_tokens"], torch.Tensor):
                d["prompt_text_tokens"] = d["prompt_text_tokens"].tolist()
            data[pid] = d
            
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def add(self, name: str, speaker_embed: torch.Tensor, prompt_codec_tokens: torch.Tensor | None = None) -> SpeakerProfile:
        profile = SpeakerProfile(
            speaker_profile_id=uuid.uuid4().hex[:8],
            display_name=name,
            speaker_embed=speaker_embed,
            prompt_codec_tokens=prompt_codec_tokens,
            reference_audio_hash="sha256_placeholder",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self.profiles[profile.speaker_profile_id] = profile
        self.save()
        return profile

    def remove(self, profile_id: str) -> None:
        self.profiles.pop(profile_id, None)
        self.save()

    def list_names(self) -> list[str]:
        return [f"{p.name} ({pid})" for pid, p in self.profiles.items()]


# ---------------------------------------------------------------------------
# Evaluation Session
# ---------------------------------------------------------------------------

@dataclass
class EvalPair:
    pair_id: str
    sample_a_label: str
    sample_b_label: str
    text: str
    preference: str = ""  # "A", "B", "tie", ""
    mos_a: float = 0.0
    mos_b: float = 0.0
    rater_id: str = ""
    rater_role: str = ""
    timestamp: float = 0.0
    notes: str = ""


class EvalSession:
    """Stores blind A/B evaluation results."""

    def __init__(self, path: Path | str = "data/eval_sessions.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, pair: EvalPair) -> None:
        pair.timestamp = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines]

    def summary(self) -> dict:
        records = self.read_all()
        if not records:
            return {"total": 0, "a_wins": 0, "b_wins": 0, "ties": 0}
        a_wins = sum(1 for r in records if r.get("preference") == "A")
        b_wins = sum(1 for r in records if r.get("preference") == "B")
        ties = sum(1 for r in records if r.get("preference") == "tie")
        return {
            "total": len(records),
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
        }
