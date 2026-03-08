"""Run control-parameter sweep evaluation (evaluation-set-spec SS3.3).

Reads a manifest.jsonl, filters to subset="control_sweeps", and for each
item sweeps 5 frozen levels of the relevant control parameter (pace,
hold_bias, or boundary_bias) via POST /tts on the TMRVC serve API.

Total renders: 9 languages x 3 prompts x 5 levels = 135.

Usage:
    python scripts/eval/run_control_sweeps.py \
        --manifest eval/manifest.jsonl \
        --output-dir eval/control_sweep_outputs/ \
        --character-id default_narrator
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGES = ("zh", "en", "ja", "ko", "de", "fr", "ru", "es", "it")

SWEEP_LEVELS: dict[str, list[float]] = {
    "pace": [0.85, 0.95, 1.00, 1.05, 1.15],
    "hold_bias": [-0.5, -0.25, 0.0, 0.25, 0.5],
    "boundary_bias": [-0.5, -0.25, 0.0, 0.25, 0.5],
}

CONTROL_PARAMS = tuple(SWEEP_LEVELS.keys())

EXPECTED_BASE_PROMPTS = len(LANGUAGES) * len(CONTROL_PARAMS)  # 27
LEVELS_PER_PARAM = 5
EXPECTED_TOTAL_RENDERS = EXPECTED_BASE_PROMPTS * LEVELS_PER_PARAM  # 135

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_control_param(item: dict[str, Any]) -> str:
    """Determine which control parameter an item tests.

    Checks the manifest ``control_param`` field first, then falls back to
    matching the ``item_id`` against known parameter names.
    """
    # Explicit field
    if "control_param" in item:
        param = item["control_param"]
        if param in SWEEP_LEVELS:
            return param
        raise ValueError(
            f"Unknown control_param {param!r} in item {item.get('item_id')}"
        )

    # Convention: item_id contains the parameter name
    item_id: str = item.get("item_id", "")
    for param in CONTROL_PARAMS:
        if param in item_id:
            return param

    raise ValueError(
        f"Cannot determine control parameter for item {item_id!r}. "
        "Set 'control_param' in the manifest or use a convention like "
        "'zh_pace_001' in item_id."
    )


def _build_tts_payload(
    text: str,
    character_id: str,
    param: str,
    level: float,
) -> dict[str, Any]:
    """Build the JSON payload for POST /tts."""
    payload: dict[str, Any] = {
        "text": text,
        "character_id": character_id,
    }
    # Set the swept parameter; leave others at default
    payload[param] = level
    return payload


def _save_audio(
    audio_b64: str,
    meta: dict[str, Any],
    output_dir: Path,
    stem: str,
) -> Path:
    """Decode base64 WAV and write audio + metadata JSON."""
    audio_path = output_dir / f"{stem}.wav"
    audio_path.write_bytes(base64.b64decode(audio_b64))

    meta_path = output_dir / f"{stem}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return audio_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_sweeps(
    manifest_path: Path,
    api_url: str,
    output_dir: Path,
    character_id: str,
    timeout: int,
) -> None:
    """Execute the control sweep evaluation."""
    # ------------------------------------------------------------------
    # 1. Read manifest, filter to control_sweeps subset
    # ------------------------------------------------------------------
    items: list[dict[str, Any]] = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("subset") == "control_sweeps":
                items.append(row)

    if not items:
        logger.error(
            "No items with subset='control_sweeps' found in %s", manifest_path
        )
        return

    logger.info(
        "Loaded %d control_sweeps items from %s (expected %d)",
        len(items),
        manifest_path,
        EXPECTED_BASE_PROMPTS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    tts_endpoint = f"{api_url.rstrip('/')}/tts"

    # ------------------------------------------------------------------
    # 2. Sweep
    # ------------------------------------------------------------------
    total_renders = 0
    failures: list[dict[str, Any]] = []

    for item in items:
        item_id: str = item["item_id"]
        text: str = item["target_text"]
        param = _detect_control_param(item)
        levels = SWEEP_LEVELS[param]

        for level in levels:
            stem = f"{item_id}__{param}_{level}"
            logger.info("Rendering %s", stem)

            payload = _build_tts_payload(text, character_id, param, level)

            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    tts_endpoint, json=payload, timeout=timeout
                )
                elapsed = time.perf_counter() - t0
                resp.raise_for_status()
            except Exception as exc:
                logger.error("FAILED %s: %s", stem, exc)
                failures.append({"stem": stem, "error": str(exc)})
                continue

            body = resp.json()
            audio_b64 = body.get("audio_base64", "")
            if not audio_b64:
                logger.error("FAILED %s: empty audio_base64 in response", stem)
                failures.append({"stem": stem, "error": "empty audio_base64"})
                continue

            meta: dict[str, Any] = {
                "item_id": item_id,
                "control_param": param,
                "level": level,
                "character_id": character_id,
                "text": text,
                "language_id": item.get("language_id"),
                "sample_rate": body.get("sample_rate", 24000),
                "duration_sec": body.get("duration_sec", 0.0),
                "request_elapsed_sec": round(elapsed, 4),
                "audio_file": f"{stem}.wav",
            }

            audio_path = _save_audio(audio_b64, meta, output_dir, stem)
            total_renders += 1
            logger.info(
                "  -> %s (%.2fs audio, %.3fs request)",
                audio_path,
                meta["duration_sec"],
                elapsed,
            )

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    logger.info("---")
    logger.info("Total renders: %d / %d expected", total_renders, EXPECTED_TOTAL_RENDERS)
    if failures:
        logger.warning("Failures: %d", len(failures))
        for f in failures:
            logger.warning("  %s: %s", f["stem"], f["error"])
    else:
        logger.info("All renders succeeded.")
    logger.info("Outputs in %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run control-parameter sweep evaluation (spec SS3.3)."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.jsonl. Items with subset='control_sweeps' are used.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"TMRVC serve endpoint (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where rendered audio and metadata are saved.",
    )
    parser.add_argument(
        "--character-id",
        required=True,
        help="Character ID to use for TTS requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    run_sweeps(
        manifest_path=Path(args.manifest),
        api_url=args.api_url,
        output_dir=Path(args.output_dir),
        character_id=args.character_id,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
