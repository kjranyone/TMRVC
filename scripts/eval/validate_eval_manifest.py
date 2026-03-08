"""Validate a frozen evaluation manifest.jsonl against the evaluation-set spec.

Usage:
    python scripts/eval/validate_eval_manifest.py --manifest path/to/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Frozen spec constants
# ---------------------------------------------------------------------------

EXPECTED_SUBSETS: dict[str, int] = {
    "read_core": 27,
    "dialogue_context_pairs": 36,
    "control_sweeps": 135,
    "few_shot_same_language": 108,
    "few_shot_leakage_pairs": 54,
    "code_switch_probe": 12,
}

SIGN_OFF_LANGUAGES = ["zh", "en", "ja", "ko", "de", "fr", "ru", "es", "it"]

REFERENCE_LENGTHS_SEC = [3, 5, 10]

HUMAN_EVAL_ARMS: dict[str, int] = {
    "ab_primary_quality": 54,
    "ab_secondary_streaming": 36,
    "ab_v2_regression": 27,
    "mos_primary": 36,
}

# ---------------------------------------------------------------------------
# Required fields per subset
# ---------------------------------------------------------------------------

BASE_REQUIRED_FIELDS = [
    "item_id",
    "subset",
    "language_id",
    "target_text",
    "target_text_norm",
    "human_eval_eligible",
    "automated_eval_eligible",
    "notes",
]

DIALOGUE_EXTRA_FIELDS = [
    "pair_id",
    "context_variant_id",
    "dialogue_context_id",
    "dialogue_context_text",
]

FEW_SHOT_EXTRA_FIELDS = [
    "speaker_id",
    "reference_audio_id",
    "reference_text",
    "reference_length_sec",
    "reference_session_id",
    "target_session_id",
    "cross_utterance_verified",
]

CODE_SWITCH_EXTRA_FIELDS = [
    "code_switch_spans",
]

FEW_SHOT_SUBSETS = {"few_shot_same_language", "few_shot_leakage_pairs"}


def _required_fields_for(subset: str) -> list[str]:
    fields = list(BASE_REQUIRED_FIELDS)
    if subset == "dialogue_context_pairs":
        fields.extend(DIALOGUE_EXTRA_FIELDS)
    elif subset in FEW_SHOT_SUBSETS:
        fields.extend(FEW_SHOT_EXTRA_FIELDS)
    elif subset == "code_switch_probe":
        fields.extend(CODE_SWITCH_EXTRA_FIELDS)
    return fields


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_manifest(path: Path) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    rows: list[dict] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                results.append(("json_parse", False, f"Line {lineno}: {exc}"))
                return results

    if not rows:
        results.append(("non_empty", False, "Manifest is empty"))
        return results

    # 1. Required fields per subset
    missing_fields_errors: list[str] = []
    for i, row in enumerate(rows):
        subset = row.get("subset", "<missing>")
        required = _required_fields_for(subset)
        missing = [f for f in required if f not in row]
        if missing:
            item = row.get("item_id", f"row_{i}")
            missing_fields_errors.append(f"{item}: missing {missing}")
    if missing_fields_errors:
        results.append(("required_fields", False, "; ".join(missing_fields_errors[:10])))
    else:
        results.append(("required_fields", True, "all rows have required fields"))

    # 2. Subset item counts
    subset_counts = Counter(row.get("subset") for row in rows)
    count_errors: list[str] = []
    for subset, expected in EXPECTED_SUBSETS.items():
        actual = subset_counts.get(subset, 0)
        if actual != expected:
            count_errors.append(f"{subset}: expected {expected}, got {actual}")
    unexpected = set(subset_counts) - set(EXPECTED_SUBSETS)
    for s in unexpected:
        count_errors.append(f"unexpected subset: {s}")
    if count_errors:
        results.append(("subset_counts", False, "; ".join(count_errors)))
    else:
        results.append(("subset_counts", True, "all subset counts match"))

    # 3. Language IDs
    bad_langs = {
        row.get("language_id")
        for row in rows
        if row.get("language_id") not in SIGN_OFF_LANGUAGES
    }
    if bad_langs:
        results.append(("language_ids", False, f"invalid languages: {sorted(bad_langs)}"))
    else:
        results.append(("language_ids", True, "all language_ids valid"))

    # 4. Few-shot reference_length_sec
    bad_ref_lens: list[str] = []
    for row in rows:
        if row.get("subset") in FEW_SHOT_SUBSETS:
            ref_len = row.get("reference_length_sec")
            if ref_len not in REFERENCE_LENGTHS_SEC:
                bad_ref_lens.append(f"{row.get('item_id')}: {ref_len}")
    if bad_ref_lens:
        results.append(("reference_length_sec", False, "; ".join(bad_ref_lens[:10])))
    else:
        results.append(("reference_length_sec", True, "all reference lengths valid"))

    # 5. Few-shot cross_utterance_verified
    unverified: list[str] = []
    for row in rows:
        if row.get("subset") in FEW_SHOT_SUBSETS:
            if row.get("cross_utterance_verified") is not True:
                unverified.append(str(row.get("item_id")))
    if unverified:
        results.append(("cross_utterance_verified", False, f"not True: {unverified[:10]}"))
    else:
        results.append(("cross_utterance_verified", True, "all few-shot items verified"))

    # 6. Dialogue pair_id has exactly 2 variants
    pair_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("subset") == "dialogue_context_pairs":
            pid = row.get("pair_id")
            if pid is not None:
                pair_counts[pid] += 1
    bad_pairs = {pid: cnt for pid, cnt in pair_counts.items() if cnt != 2}
    if bad_pairs:
        results.append(("dialogue_pairs", False, f"pair_ids without exactly 2 variants: {bad_pairs}"))
    else:
        results.append(("dialogue_pairs", True, "all pair_ids have exactly 2 variants"))

    # 7. No duplicate item_ids
    id_counts = Counter(row.get("item_id") for row in rows)
    dupes = {k: v for k, v in id_counts.items() if v > 1}
    if dupes:
        results.append(("unique_item_ids", False, f"duplicate item_ids: {dupes}"))
    else:
        results.append(("unique_item_ids", True, "all item_ids unique"))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an evaluation manifest.jsonl against the frozen spec."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.jsonl file.",
    )
    args = parser.parse_args()

    path = Path(args.manifest)
    if not path.exists():
        print(f"FAIL: manifest file not found: {path}")
        sys.exit(1)

    results = validate_manifest(path)

    any_fail = False
    for check_name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            any_fail = True
        print(f"  [{status}] {check_name}: {detail}")

    if any_fail:
        print("\nValidation FAILED.")
        sys.exit(1)
    else:
        print("\nAll checks PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
