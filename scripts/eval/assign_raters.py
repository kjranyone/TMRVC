"""Human evaluation assignment engine.

Reads a frozen manifest.jsonl, constructs the four human-eval arms defined in
docs/design/evaluation-set-spec.md section 5, injects duplicate QC items, and
assigns items to raters with balanced language / presentation-order coverage.

Usage:
    python scripts/eval/assign_raters.py \
        --manifest eval/sets/tmrvc_eval_public_v1_2026_03_08/manifest.jsonl \
        --num-raters 30 \
        --output assignments.json \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants from evaluation-set-spec.md section 5
# ---------------------------------------------------------------------------

ARM_SPECS: dict[str, dict[str, int]] = {
    "ab_primary_quality": {
        "read_core": 18,
        "dialogue_context_pairs": 18,
        "few_shot_same_language": 18,
    },
    "ab_secondary_streaming": {
        "dialogue_context_pairs": 18,
        "few_shot_same_language": 18,
    },
    "ab_v2_regression": {
        "read_core": 9,
        "dialogue_context_pairs": 9,
        "few_shot_same_language": 9,
    },
    "mos_primary": {
        "read_core": 12,
        "dialogue_context_pairs": 12,
        "few_shot_same_language": 12,
    },
}

ARM_TOTALS: dict[str, int] = {
    "ab_primary_quality": 54,
    "ab_secondary_streaming": 36,
    "ab_v2_regression": 27,
    "mos_primary": 36,
}

PAIRWISE_ARMS = {"ab_primary_quality", "ab_secondary_streaming", "ab_v2_regression"}
MOS_ARMS = {"mos_primary"}

MIN_UNIQUE_RATERS = 30
MIN_RATINGS_PAIRWISE = 6
MIN_RATINGS_MOS = 8
DUPLICATE_RATIO = 0.125
WORKLOAD_MIN = 32
WORKLOAD_MAX = 40


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load manifest.jsonl and return rows marked human_eval_eligible."""
    items: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("human_eval_eligible"):
                items.append(row)
    logger.info("Loaded %d human-eval-eligible items from %s", len(items), path)
    return items


# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------

def _select_items(
    pool: list[dict[str, Any]],
    subset: str,
    count: int,
    rng: random.Random,
    already_used: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select *count* items from *pool* matching *subset*, balanced by language.

    Items whose item_id appears in *already_used* are deprioritised but will
    still be picked if the pool is too small.
    """
    candidates = [r for r in pool if r["subset"] == subset]
    if already_used:
        preferred = [c for c in candidates if c["item_id"] not in already_used]
        fallback = [c for c in candidates if c["item_id"] in already_used]
    else:
        preferred = candidates
        fallback = []

    # Language-balanced selection: round-robin across languages
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in preferred:
        by_lang[item["language_id"]].append(item)
    for lang in by_lang:
        rng.shuffle(by_lang[lang])

    selected: list[dict[str, Any]] = []
    langs = sorted(by_lang.keys())
    idx = 0
    while len(selected) < count and any(by_lang[l] for l in langs):
        lang = langs[idx % len(langs)]
        if by_lang[lang]:
            selected.append(by_lang[lang].pop(0))
        idx += 1

    # If we still need more, dip into fallback
    if len(selected) < count:
        rng.shuffle(fallback)
        for item in fallback:
            if len(selected) >= count:
                break
            selected.append(item)

    if len(selected) < count:
        logger.warning(
            "Could only select %d/%d items for subset=%s",
            len(selected),
            count,
            subset,
        )

    return selected[:count]


def build_arms(
    items: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    """Build the four evaluation arms from eligible manifest items."""
    arms: dict[str, list[dict[str, Any]]] = {}
    global_used: set[str] = set()

    for arm_name, subset_counts in ARM_SPECS.items():
        arm_items: list[dict[str, Any]] = []
        for subset, count in subset_counts.items():
            selected = _select_items(items, subset, count, rng, global_used)
            arm_items.extend(selected)
        rng.shuffle(arm_items)
        arms[arm_name] = arm_items
        for it in arm_items:
            global_used.add(it["item_id"])

    for arm_name, arm_items in arms.items():
        expected = ARM_TOTALS[arm_name]
        if len(arm_items) != expected:
            logger.warning(
                "Arm %s has %d items (expected %d)", arm_name, len(arm_items), expected
            )

    return arms


# ---------------------------------------------------------------------------
# Duplicate injection
# ---------------------------------------------------------------------------

def inject_duplicates(
    arm_items: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Add duplicate QC items at the configured ratio.

    Returns a new list containing both original and duplicate entries.
    Duplicate entries get an ``is_duplicate`` flag set to True.
    """
    n_dups = max(1, round(len(arm_items) * DUPLICATE_RATIO))
    dup_sources = rng.sample(arm_items, min(n_dups, len(arm_items)))

    augmented = []
    for item in arm_items:
        augmented.append({**item, "is_duplicate": False})
    for item in dup_sources:
        augmented.append({**item, "is_duplicate": True})

    rng.shuffle(augmented)
    return augmented


# ---------------------------------------------------------------------------
# Assignment engine
# ---------------------------------------------------------------------------

def _min_ratings_for_arm(arm_name: str) -> int:
    if arm_name in PAIRWISE_ARMS:
        return MIN_RATINGS_PAIRWISE
    return MIN_RATINGS_MOS


def assign_items_to_raters(
    arms: dict[str, list[dict[str, Any]]],
    num_raters: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Assign arm items to raters respecting all constraints.

    Returns a list of rater assignment dicts.
    """
    # Inject duplicates into each arm
    arm_pools: dict[str, list[dict[str, Any]]] = {}
    for arm_name, items in arms.items():
        arm_pools[arm_name] = inject_duplicates(items, rng)

    # Build the assignment data structure
    # For each (arm, item_index) we track which raters have been assigned.
    # item_index is position in the arm_pools list (unique per duplicate too).
    RaterAssignment = dict[str, list[dict[str, Any]]]  # rater_id -> items
    assignments: RaterAssignment = {
        f"rater_{i + 1:03d}": [] for i in range(num_raters)
    }

    # Track per-rater: which item_ids they have seen (across all arms)
    rater_seen_ids: dict[str, set[str]] = defaultdict(set)
    # Track per-rater: language counts for balancing
    rater_lang_counts: dict[str, Counter] = defaultdict(Counter)
    # Track per-item: how many ratings it has received
    item_rating_counts: dict[str, int] = defaultdict(int)
    # Unique key per pool entry to allow duplicate items to be tracked separately
    _entry_key_counter = 0

    # Flatten all items with metadata
    all_entries: list[dict[str, Any]] = []
    for arm_name, pool in arm_pools.items():
        min_ratings = _min_ratings_for_arm(arm_name)
        for idx, entry in enumerate(pool):
            _entry_key_counter += 1
            all_entries.append({
                "entry_key": f"{arm_name}:{_entry_key_counter}",
                "arm": arm_name,
                "item_id": entry["item_id"],
                "language_id": entry.get("language_id", "unknown"),
                "is_duplicate": entry.get("is_duplicate", False),
                "min_ratings": min_ratings,
                "assigned_count": 0,
            })

    # Greedy assignment: iterate until every entry has enough ratings
    # Sort entries by how far they are from their minimum, ascending.
    max_iterations = 200
    for iteration in range(max_iterations):
        # Find entries that still need ratings
        needy = [e for e in all_entries if e["assigned_count"] < e["min_ratings"]]
        if not needy:
            break

        # Sort by deficit (most-needed first) for better coverage
        needy.sort(key=lambda e: e["assigned_count"] - e["min_ratings"])

        progress_made = False
        for entry in needy:
            if entry["assigned_count"] >= entry["min_ratings"]:
                continue

            # Find eligible raters: those who haven't seen this item_id in
            # another arm, and whose workload is below max.
            eligible_raters = []
            for rater_id in assignments:
                workload = len(assignments[rater_id])
                if workload >= WORKLOAD_MAX:
                    continue
                # Check cross-arm conflict: rater must not see same item_id
                # in a different arm
                if entry["item_id"] in rater_seen_ids[rater_id]:
                    # Allow if the existing assignment is in the SAME arm
                    same_arm = any(
                        a["item_id"] == entry["item_id"] and a["arm"] == entry["arm"]
                        for a in assignments[rater_id]
                    )
                    if not same_arm:
                        continue
                # Check rater hasn't already been assigned this exact entry
                if any(
                    a.get("entry_key") == entry["entry_key"]
                    for a in assignments[rater_id]
                ):
                    continue
                eligible_raters.append(rater_id)

            if not eligible_raters:
                continue

            # Pick rater with best balance: lowest workload first,
            # then lowest count for this entry's language
            eligible_raters.sort(
                key=lambda r: (
                    len(assignments[r]),
                    rater_lang_counts[r][entry["language_id"]],
                )
            )
            chosen = eligible_raters[0]

            # Randomize presentation order for pairwise arms
            if entry["arm"] in PAIRWISE_ARMS:
                presentation_order = rng.choice(["AB", "BA"])
            else:
                presentation_order = None

            record = {
                "item_id": entry["item_id"],
                "arm": entry["arm"],
                "is_duplicate": entry["is_duplicate"],
                "entry_key": entry["entry_key"],
            }
            if presentation_order is not None:
                record["presentation_order"] = presentation_order

            assignments[chosen].append(record)
            rater_seen_ids[chosen].add(entry["item_id"])
            rater_lang_counts[chosen][entry["language_id"]] += 1
            entry["assigned_count"] += 1
            progress_made = True

        if not progress_made:
            logger.warning(
                "Assignment stalled at iteration %d with %d entries still needy",
                iteration,
                len([e for e in all_entries if e["assigned_count"] < e["min_ratings"]]),
            )
            break

    # Pad raters below WORKLOAD_MIN by giving them extra items from arms
    # that benefit from additional ratings
    for rater_id in assignments:
        while len(assignments[rater_id]) < WORKLOAD_MIN:
            # Find an entry this rater can still take
            found = False
            candidates = sorted(
                all_entries,
                key=lambda e: e["assigned_count"],
            )
            for entry in candidates:
                if entry["item_id"] in rater_seen_ids[rater_id]:
                    same_arm = any(
                        a["item_id"] == entry["item_id"] and a["arm"] == entry["arm"]
                        for a in assignments[rater_id]
                    )
                    if not same_arm:
                        continue
                if any(
                    a.get("entry_key") == entry["entry_key"]
                    for a in assignments[rater_id]
                ):
                    continue

                if entry["arm"] in PAIRWISE_ARMS:
                    presentation_order = rng.choice(["AB", "BA"])
                else:
                    presentation_order = None

                record = {
                    "item_id": entry["item_id"],
                    "arm": entry["arm"],
                    "is_duplicate": entry["is_duplicate"],
                    "entry_key": entry["entry_key"],
                }
                if presentation_order is not None:
                    record["presentation_order"] = presentation_order

                assignments[rater_id].append(record)
                rater_seen_ids[rater_id].add(entry["item_id"])
                rater_lang_counts[rater_id][entry["language_id"]] += 1
                entry["assigned_count"] += 1
                found = True
                break
            if not found:
                break

    # Build output structure (strip internal keys)
    result = []
    for rater_id in sorted(assignments):
        rater_items = []
        for a in assignments[rater_id]:
            item: dict[str, Any] = {
                "item_id": a["item_id"],
                "arm": a["arm"],
                "is_duplicate": a["is_duplicate"],
            }
            if "presentation_order" in a:
                item["presentation_order"] = a["presentation_order"]
            rater_items.append(item)
        result.append({
            "rater_id": rater_id,
            "items": rater_items,
        })
    return result


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def compute_coverage(
    assignments: list[dict[str, Any]],
    arms: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compute and return a coverage summary dict."""
    # Per-arm, per-item rating count
    item_counts: dict[str, Counter] = defaultdict(Counter)
    for rater in assignments:
        for item in rater["items"]:
            item_counts[item["arm"]][item["item_id"]] += 1

    arm_reports: dict[str, Any] = {}
    for arm_name in ARM_SPECS:
        counts = item_counts.get(arm_name, Counter())
        if counts:
            min_c = min(counts.values())
            max_c = max(counts.values())
            mean_c = sum(counts.values()) / len(counts)
        else:
            min_c = max_c = 0
            mean_c = 0.0
        target = _min_ratings_for_arm(arm_name)
        items_below = sum(1 for c in counts.values() if c < target)
        arm_reports[arm_name] = {
            "num_items": len(counts),
            "min_ratings": min_c,
            "max_ratings": max_c,
            "mean_ratings": round(mean_c, 2),
            "target_min_ratings": target,
            "items_below_target": items_below,
        }

    workloads = [len(r["items"]) for r in assignments]
    raters_in_range = sum(1 for w in workloads if WORKLOAD_MIN <= w <= WORKLOAD_MAX)

    # Presentation order balance (pairwise arms only)
    order_counts: Counter = Counter()
    for rater in assignments:
        for item in rater["items"]:
            if "presentation_order" in item:
                order_counts[item["presentation_order"]] += 1

    return {
        "num_raters": len(assignments),
        "workload_min": min(workloads) if workloads else 0,
        "workload_max": max(workloads) if workloads else 0,
        "workload_mean": round(sum(workloads) / len(workloads), 2) if workloads else 0,
        "raters_in_target_range": raters_in_range,
        "presentation_order_balance": dict(order_counts),
        "arms": arm_reports,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human evaluation assignment engine (evaluation-set-spec.md section 5)."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to manifest.jsonl.",
    )
    parser.add_argument(
        "--num-raters",
        type=int,
        default=MIN_UNIQUE_RATERS,
        help=f"Number of raters (default: {MIN_UNIQUE_RATERS}).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output assignment JSON path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    rng = random.Random(args.seed)

    # 1. Load manifest
    items = load_manifest(args.manifest)
    if not items:
        logger.error("No human_eval_eligible items found in %s", args.manifest)
        raise SystemExit(1)

    # 2. Build arms
    arms = build_arms(items, rng)
    for arm_name, arm_items in arms.items():
        logger.info(
            "Arm %-25s: %d items", arm_name, len(arm_items)
        )

    # 3. Assign to raters
    assignments = assign_items_to_raters(arms, args.num_raters, rng)

    # 4. Coverage report
    coverage = compute_coverage(assignments, arms)

    # 5. Build output
    arms_summary: dict[str, Any] = {}
    for arm_name, arm_items in arms.items():
        lang_dist: Counter = Counter()
        subset_dist: Counter = Counter()
        for it in arm_items:
            lang_dist[it.get("language_id", "unknown")] += 1
            subset_dist[it.get("subset", "unknown")] += 1
        arms_summary[arm_name] = {
            "total_items": len(arm_items),
            "language_distribution": dict(lang_dist),
            "subset_distribution": dict(subset_dist),
        }

    output = {
        "seed": args.seed,
        "num_raters": args.num_raters,
        "arms": arms_summary,
        "assignments": assignments,
        "coverage_report": coverage,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Wrote assignments to %s", args.output)

    # 6. Print summary
    print("\n=== Coverage Summary ===")
    print(f"Raters:          {coverage['num_raters']}")
    print(f"Workload range:  {coverage['workload_min']}-{coverage['workload_max']} "
          f"(mean {coverage['workload_mean']})")
    print(f"In target range: {coverage['raters_in_target_range']}/{coverage['num_raters']}")
    print(f"A/B order:       {coverage['presentation_order_balance']}")
    print()
    for arm_name, report in coverage["arms"].items():
        status = "OK" if report["items_below_target"] == 0 else "BELOW"
        print(
            f"  {arm_name:25s}  items={report['num_items']:3d}  "
            f"ratings={report['min_ratings']}-{report['max_ratings']} "
            f"(mean {report['mean_ratings']})  "
            f"target>={report['target_min_ratings']}  [{status}]"
        )
    print()


if __name__ == "__main__":
    main()
