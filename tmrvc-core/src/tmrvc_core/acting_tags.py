"""Frozen inline acting tag vocabulary for v4.

Acting tags are embedded inline in enriched transcripts alongside phoneme sequences.
They serve as a complementary text-conditioned acting path alongside physical controls
and acting texture latent.

Tag categories:
- vocal events: [inhale], [exhale], [laugh], [sigh], [cough], [click]
- prosodic markers: [emphasis], [prolonged], [pause]
- acting directives: [angry], [whisper], [calm], [excited], [tender], [professional]
- free-form: [in a hurry], [with a slight smile], etc. -> normalized by annotation LLM

Rules:
- Tags are consumed by the pointer as text units (not skipped)
- Tags co-exist with physical controls and acting latent
- If both an inline tag and explicit physical override exist, physical override takes precedence
- Tag embedding dimension matches text encoder hidden dimension
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple


# ---------------------------------------------------------------------------
# Frozen Tag Vocabulary
# ---------------------------------------------------------------------------

VOCAL_EVENT_TAGS: Tuple[str, ...] = (
    "[inhale]",
    "[exhale]",
    "[laugh]",
    "[sigh]",
    "[cough]",
    "[click]",
    "[gasp]",
    "[hum]",
    "[cry]",
    "[sniff]",
)

PROSODIC_MARKER_TAGS: Tuple[str, ...] = (
    "[emphasis]",
    "[prolonged]",
    "[pause]",
    "[rising]",
    "[falling]",
    "[break]",
)

ACTING_DIRECTIVE_TAGS: Tuple[str, ...] = (
    "[angry]",
    "[whisper]",
    "[calm]",
    "[excited]",
    "[tender]",
    "[professional]",
    "[sad]",
    "[happy]",
    "[fearful]",
    "[disgusted]",
    "[surprised]",
    "[bored]",
    "[nervous]",
    "[confident]",
    "[sarcastic]",
    "[playful]",
)

# Special token for free-form acting instructions
# The text inside brackets is embedded via a learned projection
FREEFORM_OPEN_TAG = "[act:"
FREEFORM_CLOSE_TAG = "]"

# All frozen tags
ALL_ACTING_TAGS: Tuple[str, ...] = (
    *VOCAL_EVENT_TAGS,
    *PROSODIC_MARKER_TAGS,
    *ACTING_DIRECTIVE_TAGS,
)

# Tag to ID mapping (appended after phoneme vocab)
# These IDs start AFTER the phoneme vocabulary
def build_acting_tag_vocab(phoneme_vocab_size: int = 200) -> Dict[str, int]:
    """Build tag-to-ID mapping, starting after the phoneme vocabulary.

    Args:
        phoneme_vocab_size: Size of the base phoneme vocabulary.

    Returns:
        Dict mapping tag string to integer ID.
    """
    tag_vocab = {}
    offset = phoneme_vocab_size

    # Reserved special acting tokens
    tag_vocab["[act_start]"] = offset
    tag_vocab["[act_end]"] = offset + 1
    offset += 2

    # Fixed vocabulary tags
    for tag in ALL_ACTING_TAGS:
        tag_vocab[tag] = offset
        offset += 1

    # Free-form embedding token (the content is encoded separately)
    tag_vocab["[freeform]"] = offset
    offset += 1

    return tag_vocab


# Pre-built vocabulary
ACTING_TAG_VOCAB: Dict[str, int] = build_acting_tag_vocab(200)

# Extended vocabulary size (phonemes + acting tags)
EXTENDED_VOCAB_SIZE: int = 200 + len(ACTING_TAG_VOCAB)

# Tag categories for validation
TAG_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "vocal_event": VOCAL_EVENT_TAGS,
    "prosodic_marker": PROSODIC_MARKER_TAGS,
    "acting_directive": ACTING_DIRECTIVE_TAGS,
}


@dataclass(frozen=True)
class ActingTagInfo:
    """Metadata for a single acting tag occurrence in an enriched transcript."""
    tag: str
    position: int       # Index in the phoneme/tag sequence
    category: str       # vocal_event, prosodic_marker, acting_directive, freeform
    freeform_text: str = ""  # Only for freeform tags


def parse_enriched_transcript(enriched_text: str) -> list:
    """Parse an enriched transcript into a sequence of (text_segment, tag) pairs.

    Example:
        "[inhale] hontouni [emphasis] arigatou [prolonged laugh]"
        -> [("[inhale]", "tag"), ("hontouni", "text"), ("[emphasis]", "tag"),
           ("arigatou", "text"), ("[prolonged laugh]", "freeform")]
    """
    import re

    parts = []
    pattern = r'\[([^\]]+)\]'
    last_end = 0

    for match in re.finditer(pattern, enriched_text):
        # Text before the tag
        text_before = enriched_text[last_end:match.start()].strip()
        if text_before:
            parts.append((text_before, "text"))

        tag_content = match.group(0)  # e.g., "[angry]"

        if tag_content in set(ALL_ACTING_TAGS):
            parts.append((tag_content, "tag"))
        else:
            # Free-form acting instruction
            parts.append((tag_content, "freeform"))

        last_end = match.end()

    # Remaining text after last tag
    remaining = enriched_text[last_end:].strip()
    if remaining:
        parts.append((remaining, "text"))

    return parts


def get_tag_category(tag: str) -> str:
    """Get the category of an acting tag."""
    if tag in VOCAL_EVENT_TAGS:
        return "vocal_event"
    elif tag in PROSODIC_MARKER_TAGS:
        return "prosodic_marker"
    elif tag in ACTING_DIRECTIVE_TAGS:
        return "acting_directive"
    else:
        return "freeform"
