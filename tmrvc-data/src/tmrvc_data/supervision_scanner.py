"""Dataset supervision scanner (Worker 03).

Scans a dataset cache directory and produces a machine-readable
DatasetReport covering:
- Phone inventory coverage
- unknown_phone_ratio per language
- Missing supervision artifacts (durations, voice_state, alignment)
- voice_state pseudo-label coverage
- Suprasegmental coverage
- Few-shot prompt pairing coverage
- Curation asset coverage

Usage:
    from tmrvc_data.supervision_scanner import scan_cache_dir

    report = scan_cache_dir("/path/to/cache", dataset_name="my_corpus")
    print(report.to_dict())
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from tmrvc_data.dataset_report import DatasetReport
from tmrvc_data.g2p import ID2PHONE, PHONE2ID, UNK_ID

logger = logging.getLogger(__name__)


def scan_cache_dir(
    cache_dir: str | Path,
    dataset_name: str = "",
    split: str = "train",
) -> DatasetReport:
    """Scan a dataset cache directory and produce a DatasetReport.

    Args:
        cache_dir: Root cache directory (contains dataset subdirs).
        dataset_name: Name for the report. If empty, uses the cache dir name.
        split: Which split to scan (default "train").

    Returns:
        Populated DatasetReport instance.
    """
    cache_dir = Path(cache_dir)
    if not dataset_name:
        dataset_name = cache_dir.name

    report = DatasetReport(dataset_name=dataset_name, split=split)

    # Counters
    total = 0
    with_text = 0
    with_canonical_text_units = 0
    with_legacy_duration = 0
    with_voice_state = 0
    with_dialogue_context = 0
    with_suprasegmentals = 0
    with_bootstrap_alignment = 0
    with_prompt_eligible = 0
    with_curation_record = 0
    total_phones = 0
    unk_phones = 0
    active_ids: set[int] = set()
    vs_observed_sum = 0.0
    vs_observed_count = 0
    vs_confidence_values: list[float] = []
    multi_take_texts: dict[str, int] = {}
    per_lang_phones: dict[str, int] = {}
    per_lang_unk: dict[str, int] = {}

    # Find all utterance directories under the cache
    utt_dirs = _find_utterance_dirs(cache_dir, split)

    for utt_dir in utt_dirs:
        meta_path = utt_dir / "meta.json"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        total += 1

        text = meta.get("text", "")
        language = meta.get("language", "unknown")

        # Text supervision
        if text:
            with_text += 1
            multi_take_texts[text] = multi_take_texts.get(text, 0) + 1

        # Canonical text units
        phoneme_ids_path = utt_dir / "phoneme_ids.npy"
        if phoneme_ids_path.exists():
            with_canonical_text_units += 1
            try:
                pids = np.load(phoneme_ids_path)
                n = len(pids)
                total_phones += n
                n_unk = int((pids == UNK_ID).sum())
                unk_phones += n_unk
                active_ids.update(int(p) for p in pids)

                # Per-language tracking
                per_lang_phones[language] = per_lang_phones.get(language, 0) + n
                per_lang_unk[language] = per_lang_unk.get(language, 0) + n_unk
            except Exception:
                pass

        # Legacy durations
        if (utt_dir / "durations.npy").exists():
            with_legacy_duration += 1

        # Suprasegmentals
        if (utt_dir / "text_suprasegmentals.npy").exists():
            with_suprasegmentals += 1

        # Bootstrap alignment
        if (utt_dir / "bootstrap_alignment.json").exists():
            with_bootstrap_alignment += 1

        # Voice state supervision
        if (utt_dir / "voice_state_targets.npy").exists():
            with_voice_state += 1
            try:
                mask_path = utt_dir / "voice_state_observed_mask.npy"
                if mask_path.exists():
                    mask = np.load(mask_path)
                    vs_observed_sum += float(mask.mean())
                    vs_observed_count += 1
                conf_path = utt_dir / "voice_state_confidence.npy"
                if conf_path.exists():
                    conf = np.load(conf_path)
                    vs_confidence_values.append(float(conf.mean()))
            except Exception:
                pass

        # Dialogue context
        if (utt_dir / "dialogue_context.npy").exists():
            with_dialogue_context += 1

        # Few-shot prompt
        if "prompt_eligible" in meta:
            with_prompt_eligible += 1

        # Curation fields
        if "curation_record_id" in meta:
            with_curation_record += 1

    # Populate report
    report.num_utterances = total
    report.text_supervision_coverage = with_text / max(total, 1)
    report.canonical_text_unit_coverage = with_canonical_text_units / max(total, 1)
    report.legacy_duration_coverage = with_legacy_duration / max(total, 1)
    report.unknown_phone_ratio = unk_phones / max(total_phones, 1)
    report.direct_hit_ratio = (total_phones - unk_phones) / max(total_phones, 1)
    report.active_phone_inventory = sorted(
        ID2PHONE.get(i, f"id:{i}") for i in active_ids if i != UNK_ID
    )
    report.dialogue_context_coverage = with_dialogue_context / max(total, 1)
    report.voice_state_supervision_coverage = with_voice_state / max(total, 1)
    report.voice_state_observed_ratio = (
        vs_observed_sum / vs_observed_count if vs_observed_count > 0 else 0.0
    )
    report.voice_state_confidence_summary = {
        "mean": sum(vs_confidence_values) / len(vs_confidence_values) if vs_confidence_values else 0.0,
        "count": float(len(vs_confidence_values)),
    }
    report.suprasegmental_coverage = with_suprasegmentals / max(total, 1)
    report.bootstrap_alignment_coverage = with_bootstrap_alignment / max(total, 1)
    report.prompt_pairing_coverage = with_prompt_eligible / max(total, 1)
    report.curation_record_coverage = with_curation_record / max(total, 1)

    multi_context_count = sum(1 for c in multi_take_texts.values() if c > 1)
    report.same_text_multi_context_coverage = (
        multi_context_count / max(len(multi_take_texts), 1) if multi_take_texts else 0.0
    )

    # Per-language stats
    for lang in sorted(set(list(per_lang_phones.keys()) + list(per_lang_unk.keys()))):
        lang_total = per_lang_phones.get(lang, 0)
        lang_unk = per_lang_unk.get(lang, 0)
        report.per_language_stats[lang] = {
            "total_phones": lang_total,
            "unk_phones": lang_unk,
            "unknown_phone_ratio": lang_unk / max(lang_total, 1),
        }

    errors = report.validate()
    if errors:
        logger.warning("DatasetReport validation errors: %s", errors)

    return report


def _find_utterance_dirs(cache_dir: Path, split: str) -> list[Path]:
    """Find all utterance directories under cache_dir/{dataset}/{split}/{speaker}/{utt}."""
    utt_dirs: list[Path] = []

    for dataset_dir in sorted(cache_dir.iterdir()) if cache_dir.exists() else []:
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("_"):
            continue
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for speaker_dir in sorted(split_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for utt_dir in sorted(speaker_dir.iterdir()):
                if utt_dir.is_dir():
                    utt_dirs.append(utt_dir)

    return utt_dirs
