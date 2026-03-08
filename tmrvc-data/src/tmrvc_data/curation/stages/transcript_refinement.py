"""Stage 5: Transcript Refinement - Multi-ASR fusion and text normalization.

Converts one-pass pseudo-labels into stable pseudo-supervision by:
- Computing agreement across multiple ASR outputs (when available)
- Selecting or fusing the best transcript
- Running text normalization
- Recording disagreement metrics and correction provenance
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from ..models import CurationRecord, Provenance, RecordStatus
from ..providers import ProviderRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalization utilities
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form."""
    return unicodedata.normalize("NFC", text)


def _normalize_punctuation(text: str) -> str:
    """Basic punctuation normalization for transcript text."""
    # Normalize various quote types to standard quotes
    text = re.sub(r"[\u2018\u2019\u0060]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    # Normalize various dashes
    text = re.sub(r"[\u2013\u2014]", "-", text)
    # Remove repeated punctuation (keep at most 1)
    text = re.sub(r"([.!?])\1+", r"\1", text)
    return text


def normalize_transcript(text: str) -> str:
    """Apply full text normalization pipeline to transcript."""
    text = _normalize_unicode(text)
    text = _normalize_punctuation(text)
    text = _normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Multi-ASR agreement computation
# ---------------------------------------------------------------------------

def _tokenize_simple(text: str) -> List[str]:
    """Simple word-level tokenizer for agreement computation."""
    return text.lower().split()


def _compute_word_overlap(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute word-level overlap ratio (Jaccard-like) between two token lists."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _compute_edit_distance_ratio(a: str, b: str) -> float:
    """Compute normalized edit distance ratio (0=identical, 1=completely different)."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0

    # Simple Levenshtein at word level for efficiency
    words_a = a.lower().split()
    words_b = b.lower().split()
    m, n = len(words_a), len(words_b)

    if m == 0 or n == 0:
        return 1.0

    # Use two-row DP for memory efficiency
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if words_a[i - 1] == words_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    distance = prev[n]
    max_len = max(m, n)
    return distance / max_len if max_len > 0 else 0.0


def compute_multi_asr_agreement(
    asr_outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute agreement metrics across multiple ASR outputs.

    Args:
        asr_outputs: List of dicts with 'text', 'confidence', 'provider' keys.

    Returns:
        Dict with agreement metrics and best transcript selection.
    """
    if not asr_outputs:
        return {
            "best_text": "",
            "best_confidence": 0.0,
            "agreement_ratio": 0.0,
            "n_sources": 0,
            "disagreement_metrics": {},
        }

    if len(asr_outputs) == 1:
        out = asr_outputs[0]
        return {
            "best_text": out.get("text", ""),
            "best_confidence": out.get("confidence", 0.0),
            "agreement_ratio": 1.0,
            "n_sources": 1,
            "disagreement_metrics": {"single_source": True},
        }

    # Select best by confidence
    best = max(asr_outputs, key=lambda x: x.get("confidence", 0.0))
    best_text = best.get("text", "")
    best_conf = best.get("confidence", 0.0)

    # Compute pairwise agreement
    texts = [o.get("text", "") for o in asr_outputs]
    n = len(texts)
    pairwise_overlaps = []
    pairwise_edit_dists = []

    for i in range(n):
        for j in range(i + 1, n):
            tokens_i = _tokenize_simple(texts[i])
            tokens_j = _tokenize_simple(texts[j])
            overlap = _compute_word_overlap(tokens_i, tokens_j)
            pairwise_overlaps.append(overlap)
            edit_dist = _compute_edit_distance_ratio(texts[i], texts[j])
            pairwise_edit_dists.append(edit_dist)

    avg_overlap = float(sum(pairwise_overlaps) / len(pairwise_overlaps)) if pairwise_overlaps else 0.0
    avg_edit_dist = float(sum(pairwise_edit_dists) / len(pairwise_edit_dists)) if pairwise_edit_dists else 0.0

    # Exact match ratio
    exact_matches = sum(1 for t in texts if t == best_text)
    exact_ratio = exact_matches / n

    # Agreement is a blend of exact match and word overlap
    agreement = 0.5 * exact_ratio + 0.5 * avg_overlap

    return {
        "best_text": best_text,
        "best_confidence": best_conf,
        "agreement_ratio": round(agreement, 4),
        "n_sources": n,
        "disagreement_metrics": {
            "avg_word_overlap": round(avg_overlap, 4),
            "avg_edit_distance_ratio": round(avg_edit_dist, 4),
            "exact_match_ratio": round(exact_ratio, 4),
            "providers": [o.get("provider", "unknown") for o in asr_outputs],
        },
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_transcript_refinement(
    record: CurationRecord,
    registry: Optional[ProviderRegistry] = None,
) -> Optional[CurationRecord]:
    """Process a single record through Stage 5: Transcript Refinement.

    If multiple ASR outputs exist (from Stage 4 or repeated runs), computes
    agreement and selects/fuses the best transcript. Applies text
    normalization and records correction provenance.

    Args:
        record: The curation record to process.
        registry: Optional provider registry for refinement providers.

    Returns:
        Updated CurationRecord.
    """
    if record.status == RecordStatus.REJECTED:
        return record

    # --- Try external refinement provider ---
    if registry is not None:
        provider = registry.get_primary("transcript_refinement")
        if provider is not None and provider.is_available():
            asr_outputs = record.attributes.get("asr_outputs", [])
            if asr_outputs:
                try:
                    output = provider.process(record, asr_outputs=asr_outputs)
                    for key, value in output.fields.items():
                        if key == "attributes" and isinstance(value, dict):
                            record.attributes.update(value)
                        elif hasattr(record, key):
                            setattr(record, key, value)
                    if output.provenance:
                        record.providers["transcript_refinement"] = output.provenance
                    return record
                except Exception as e:
                    logger.warning(
                        "Refinement provider failed for %s: %s",
                        record.record_id, e,
                    )

    # --- Builtin refinement ---
    asr_outputs = record.attributes.get("asr_outputs", [])

    # If no ASR outputs, use current transcript as single source
    if not asr_outputs and record.transcript:
        asr_outputs = [{
            "provider": "previous_stage",
            "text": record.transcript,
            "confidence": record.transcript_confidence or 0.0,
        }]

    # Compute multi-ASR agreement
    agreement = compute_multi_asr_agreement(asr_outputs)

    # Store original transcript for provenance
    original_transcript = record.transcript or ""

    # Apply best transcript
    best_text = agreement["best_text"]

    # Normalize the transcript
    refined_text = normalize_transcript(best_text)

    # Track what changed
    text_changed = refined_text != original_transcript
    normalization_applied = refined_text != best_text

    # Update record
    record.transcript = refined_text
    record.transcript_confidence = agreement["best_confidence"]

    # Store refinement metadata
    record.attributes["refined_transcript"] = refined_text
    record.attributes["refinement_agreement"] = agreement["agreement_ratio"]
    record.attributes["refinement_n_sources"] = agreement["n_sources"]
    record.attributes["refinement_confidence"] = agreement["agreement_ratio"]
    record.attributes["disagreement_metrics"] = agreement["disagreement_metrics"]
    record.attributes["correction_provenance"] = {
        "original_transcript": original_transcript,
        "text_changed": text_changed,
        "normalization_applied": normalization_applied,
        "method": "multi_asr_agreement" if agreement["n_sources"] > 1 else "single_source_normalization",
    }

    # If transcript is empty after refinement, flag it
    if not refined_text.strip():
        record.review_reasons.append("transcript_empty_after_refinement")

    # Low agreement is a review signal
    if agreement["n_sources"] > 1 and agreement["agreement_ratio"] < 0.5:
        record.review_reasons.append("low_asr_agreement")

    record.providers["transcript_refinement"] = Provenance(
        stage="transcript_refinement",
        provider="builtin_refiner",
        version="1.0.0",
        timestamp=time.time(),
        confidence=agreement["agreement_ratio"],
        metadata={
            "n_sources": agreement["n_sources"],
            "agreement_ratio": agreement["agreement_ratio"],
            "text_changed": text_changed,
            "normalization_applied": normalization_applied,
        },
    )

    return record
