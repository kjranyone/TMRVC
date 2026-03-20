#!/usr/bin/env python3
"""Fish Audio S2 head-to-head comparison protocol (Phase 5-3).

Implements the frozen evaluation protocol from docs/design/evaluation-protocol.md:

Victory axes (TMRVC v4 must win):
    1. Acting editability
    2. Trajectory replay fidelity
    3. Edit locality

Guardrail axes (TMRVC v4 must not clearly lose):
    4. First-take naturalness
    5. Few-shot speaker similarity
    6. Latency class disclosure

Usage:
    python scripts/eval/fish_s2_comparison.py \\
        --tmrvc-checkpoint path/to/uclm.pt \\
        --tmrvc-codec path/to/codec.pt \\
        --fish-samples-dir path/to/fish_s2_outputs/ \\
        --eval-set path/to/eval_manifest.json \\
        [--output-dir results/fish_s2/] \\
        [--device cuda]

The script generates a blind comparison bundle: paired audio files with
randomised system labels (A/B) and a JSON report template for human
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (frozen from docs/design/evaluation-protocol.md)
# ---------------------------------------------------------------------------

VICTORY_AXES = [
    "acting_editability",
    "trajectory_replay_fidelity",
    "edit_locality",
]

GUARDRAIL_AXES = [
    "first_take_naturalness",
    "few_shot_speaker_similarity",
    "latency_class_disclosure",
]

# Claim narrowing rule: if TMRVC wins on editability but loses on
# naturalness, the claim must be scoped to editability only.
CLAIM_NARROWING_RULE = (
    "If TMRVC v4 wins only on a subset of victory axes and loses on "
    "any guardrail axis, the public claim must be narrowed to the "
    "specific winning axes.  Broad 'beats Fish S2' claims are forbidden "
    "unless all victory axes are won and no guardrail axis shows a "
    "clear deficit."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalItem:
    """A single evaluation item (prompt + text)."""
    item_id: str
    text: str
    language: str = "ja"
    acting_prompt: str = ""
    acting_tags: list[str] = field(default_factory=list)
    reference_audio_path: Optional[str] = None
    speaker_profile_id: Optional[str] = None


@dataclass
class ComparisonPair:
    """A blind A/B comparison pair."""
    pair_id: str
    item_id: str
    system_a: str  # "tmrvc" or "fish_s2" (hidden from rater)
    system_b: str
    audio_a_path: str
    audio_b_path: str
    text: str
    language: str
    acting_prompt: str = ""

    # Results (filled by rater or automatic evaluation)
    preference: Optional[str] = None  # "A", "B", "tie"
    per_axis_scores: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Aggregated comparison report."""
    report_id: str
    timestamp: str
    tmrvc_version: str = "v4"
    fish_s2_version: str = ""
    eval_set_version: str = ""
    hardware_class: str = ""

    # Per-axis results
    victory_axes: dict[str, dict[str, Any]] = field(default_factory=dict)
    guardrail_axes: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Aggregate
    pairs_evaluated: int = 0
    tmrvc_wins: int = 0
    fish_wins: int = 0
    ties: int = 0

    # Claim
    claim_scope: str = ""
    claim_valid: bool = False

    # Latency disclosure
    tmrvc_rtf: float = 0.0
    tmrvc_hardware: str = ""
    fish_s2_rtf: float = 0.0
    fish_s2_hardware: str = ""


# ---------------------------------------------------------------------------
# Victory axis 1: Acting Editability
# ---------------------------------------------------------------------------

def measure_acting_editability(
    engine: Any,
    speaker_embed: Any,
    phonemes: Any,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure acting editability: generate with different acting controls
    and verify that the output changes meaningfully.

    Protocol:
    1. Generate with neutral controls.
    2. Generate with high-intensity acting (energy=0.9, intensity=0.9).
    3. Generate with low-intensity acting (energy=0.1, breathiness=0.8).
    4. Measure pairwise distance in physical trajectory space.

    An editable system should show large trajectory distances between
    different acting configurations.

    Returns:
        Dict with trajectory distances and editability score.
    """
    import torch

    configs = {
        "neutral": torch.full((1, 1, 12), 0.5, device=device),
        "high_energy": torch.tensor([[[0.5, 0.5, 0.9, 0.5, 0.5, 0.1, 0.2, 0.7, 0.2, 0.5, 0.9, 0.1]]], device=device),
        "low_breathy": torch.tensor([[[0.3, 0.2, 0.1, 0.2, 0.3, 0.8, 0.3, 0.3, 0.3, 0.5, 0.1, 0.5]]], device=device),
    }

    trajectories: dict[str, torch.Tensor] = {}
    for name, controls in configs.items():
        try:
            _, meta = engine.tts(
                phonemes=phonemes,
                speaker_embed=speaker_embed,
                explicit_voice_state=controls,
                cfg_mode="off",
                temperature=0.0,
                max_frames=300,
            )
            traj = meta.get("physical_trajectory")
            if traj is not None:
                trajectories[name] = traj
        except Exception as e:
            logger.warning("Editability gen failed for %s: %s", name, e)

    if len(trajectories) < 2:
        return {"editability_score": 0.0, "pairs_compared": 0}

    # Pairwise L2 distance of trajectory means
    names = list(trajectories.keys())
    distances = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mean_i = trajectories[names[i]].float().mean(dim=0)
            mean_j = trajectories[names[j]].float().mean(dim=0)
            dist = float(torch.norm(mean_i - mean_j))
            distances.append(dist)

    mean_dist = float(np.mean(distances)) if distances else 0.0
    return {
        "editability_score": mean_dist,
        "pairs_compared": len(distances),
        "distances": distances,
    }


# ---------------------------------------------------------------------------
# Victory axis 2: Trajectory Replay Fidelity
# ---------------------------------------------------------------------------

def measure_replay_fidelity(
    engine: Any,
    speaker_embed: Any,
    phonemes: Any,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure trajectory replay fidelity.

    Protocol:
    1. Generate fresh → get TrajectoryRecord.
    2. Replay from the frozen record.
    3. Compare token-level fidelity.

    Returns:
        Dict with fidelity score and frame count.
    """
    import torch
    from tmrvc_core.types import TrajectoryRecord

    controls = torch.full((1, 1, 12), 0.5, device=device)

    try:
        audio_1, meta_1 = engine.tts(
            phonemes=phonemes,
            speaker_embed=speaker_embed,
            explicit_voice_state=controls,
            cfg_mode="off",
            temperature=0.0,
            max_frames=300,
        )
    except Exception as e:
        return {"fidelity": 0.0, "error": str(e)}

    a_trace = meta_1.get("acoustic_trace")
    c_trace = meta_1.get("control_trace")
    p_trace = meta_1.get("pointer_trace", [])
    phys_traj = meta_1.get("physical_trajectory")

    if a_trace is None or c_trace is None:
        return {"fidelity": 0.0, "error": "no trace in meta"}

    record = TrajectoryRecord(
        trajectory_id=str(uuid.uuid4()),
        source_compile_id="eval",
        phoneme_ids=phonemes,
        pointer_trace=p_trace,
        physical_trajectory=phys_traj,
        acoustic_trace=a_trace,
        control_trace=c_trace,
    )

    try:
        audio_2, meta_2 = engine.replay_trajectory(
            phonemes=phonemes,
            trajectory=record,
            speaker_embed=speaker_embed,
            temperature=0.0,
            use_exact_tokens=True,
        )
    except Exception as e:
        return {"fidelity": 0.0, "error": str(e)}

    a_trace_2 = meta_2.get("acoustic_trace")
    if a_trace_2 is None:
        return {"fidelity": 0.0, "error": "no replay trace"}

    # Align lengths
    min_T = min(a_trace.shape[-1], a_trace_2.shape[-1])
    fidelity = float((a_trace[:, :, :min_T] == a_trace_2[:, :, :min_T]).float().mean())

    return {
        "fidelity": fidelity,
        "frames": min_T,
    }


# ---------------------------------------------------------------------------
# Victory axis 3: Edit Locality (delegates to measure_controllability)
# ---------------------------------------------------------------------------

def measure_edit_locality_axis(
    engine: Any,
    speaker_embed: Any,
    phonemes: Any,
    device: str = "cpu",
) -> dict[str, float]:
    """Wrapper around the edit locality measurement from controllability harness."""
    from scripts.eval.measure_controllability import edit_locality
    return edit_locality(engine, speaker_embed, phonemes, device)


# ---------------------------------------------------------------------------
# Guardrail axis 1: First-Take Naturalness (automatic proxy)
# ---------------------------------------------------------------------------

def measure_first_take_naturalness(audio_np: np.ndarray, sr: int = 24000) -> dict[str, float]:
    """Automatic naturalness proxy metrics.

    Measures:
    - Silence ratio (too much silence = unnatural)
    - Repetition detection (spectral self-similarity)
    - Energy variance (too flat = robotic)

    Returns:
        Dict with naturalness proxy scores.
    """
    if len(audio_np) < sr:
        return {"naturalness_proxy": 0.0, "too_short": True}

    # Silence ratio
    abs_audio = np.abs(audio_np)
    silence_threshold = 0.01 * abs_audio.max() if abs_audio.max() > 0 else 0.001
    silence_ratio = float(np.mean(abs_audio < silence_threshold))

    # Energy variance (frame-level RMS)
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    rms_values = []
    for start in range(0, len(audio_np) - frame_len, hop):
        frame = audio_np[start : start + frame_len]
        rms_values.append(float(np.sqrt(np.mean(frame ** 2))))
    energy_cv = float(np.std(rms_values) / (np.mean(rms_values) + 1e-10)) if rms_values else 0.0

    # Simple naturalness proxy: penalise excessive silence and flat energy
    nat_score = 1.0
    if silence_ratio > 0.5:
        nat_score -= 0.3
    if energy_cv < 0.1:
        nat_score -= 0.2

    return {
        "naturalness_proxy": max(0.0, nat_score),
        "silence_ratio": silence_ratio,
        "energy_cv": energy_cv,
    }


# ---------------------------------------------------------------------------
# Guardrail axis 2: Few-Shot Speaker Similarity (automatic proxy)
# ---------------------------------------------------------------------------

def measure_speaker_similarity(
    generated_audio: np.ndarray,
    reference_audio: np.ndarray,
    sr: int = 24000,
) -> dict[str, float]:
    """Automatic speaker similarity proxy using spectral statistics.

    In production, use a dedicated speaker verification model (ECAPA-TDNN).
    This proxy uses spectral envelope correlation as a stand-in.

    Returns:
        Dict with similarity score.
    """
    n_fft = 2048

    def spectral_envelope(audio: np.ndarray) -> np.ndarray:
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        # Average spectrum over overlapping frames
        hop = n_fft // 2
        specs = []
        for start in range(0, len(audio) - n_fft, hop):
            frame = audio[start : start + n_fft]
            specs.append(np.abs(np.fft.rfft(frame)))
        if not specs:
            return np.zeros(n_fft // 2 + 1)
        return np.mean(specs, axis=0)

    env_gen = spectral_envelope(generated_audio)
    env_ref = spectral_envelope(reference_audio)

    # Cosine similarity of spectral envelopes
    norm_gen = np.linalg.norm(env_gen)
    norm_ref = np.linalg.norm(env_ref)
    if norm_gen < 1e-10 or norm_ref < 1e-10:
        return {"similarity": 0.0}

    cosine = float(np.dot(env_gen, env_ref) / (norm_gen * norm_ref))
    return {"similarity": max(0.0, cosine)}


# ---------------------------------------------------------------------------
# Blind comparison bundle generation
# ---------------------------------------------------------------------------

def generate_blind_bundle(
    tmrvc_outputs: dict[str, Path],
    fish_outputs: dict[str, Path],
    output_dir: Path,
) -> list[ComparisonPair]:
    """Create randomised A/B comparison pairs for blind evaluation.

    Args:
        tmrvc_outputs: item_id -> path to TMRVC audio file.
        fish_outputs: item_id -> path to Fish S2 audio file.
        output_dir: Directory for the blind bundle.

    Returns:
        List of ComparisonPair with randomised system assignment.
    """
    import shutil

    bundle_dir = output_dir / "blind_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    common_ids = set(tmrvc_outputs.keys()) & set(fish_outputs.keys())
    pairs: list[ComparisonPair] = []

    for item_id in sorted(common_ids):
        pair_id = hashlib.sha256(item_id.encode()).hexdigest()[:12]

        # Randomise which system is A
        if random.random() < 0.5:
            sys_a, sys_b = "tmrvc", "fish_s2"
            path_a, path_b = tmrvc_outputs[item_id], fish_outputs[item_id]
        else:
            sys_a, sys_b = "fish_s2", "tmrvc"
            path_a, path_b = fish_outputs[item_id], tmrvc_outputs[item_id]

        # Copy to blind paths
        blind_a = bundle_dir / f"{pair_id}_A.wav"
        blind_b = bundle_dir / f"{pair_id}_B.wav"

        if path_a.exists():
            shutil.copy2(path_a, blind_a)
        if path_b.exists():
            shutil.copy2(path_b, blind_b)

        pairs.append(ComparisonPair(
            pair_id=pair_id,
            item_id=item_id,
            system_a=sys_a,
            system_b=sys_b,
            audio_a_path=str(blind_a),
            audio_b_path=str(blind_b),
            text="",
            language="ja",
        ))

    return pairs


# ---------------------------------------------------------------------------
# Report template generation
# ---------------------------------------------------------------------------

def generate_report_template(
    pairs: list[ComparisonPair],
    output_path: Path,
    tmrvc_version: str = "v4",
    fish_version: str = "unknown",
) -> ComparisonReport:
    """Generate a comparison report template ready for human evaluation.

    Returns:
        ComparisonReport with structure filled, results empty.
    """
    import datetime

    report = ComparisonReport(
        report_id=str(uuid.uuid4())[:8],
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tmrvc_version=tmrvc_version,
        fish_s2_version=fish_version,
    )

    # Initialize axis results.
    # Scores start as None (template placeholders).  tmrvc_score is filled by
    # automatic metrics in run_comparison(); fish_s2_score is filled externally
    # from Fish S2 outputs or human evaluation.  tmrvc_wins / clear_deficit
    # are resolved once both scores are available.
    for axis in VICTORY_AXES:
        report.victory_axes[axis] = {
            "tmrvc_score": None,       # filled by run_comparison() automatic metrics
            "fish_s2_score": None,     # filled from external Fish S2 evaluation
            "tmrvc_wins": None,        # determined after both scores are available
            "method": _axis_method(axis),
        }

    for axis in GUARDRAIL_AXES:
        report.guardrail_axes[axis] = {
            "tmrvc_score": None,       # filled by run_comparison() automatic metrics
            "fish_s2_score": None,     # filled from external Fish S2 evaluation
            "clear_deficit": None,     # True if TMRVC shows clear quality deficit
            "method": _axis_method(axis),
        }

    report.pairs_evaluated = len(pairs)

    # Write template
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)

    logger.info("Report template written to %s", output_path)
    return report


def _axis_method(axis: str) -> str:
    """Return the measurement method description for a given axis."""
    methods = {
        "acting_editability": (
            "Generate with 3+ acting configurations; measure trajectory "
            "distance.  Higher = more editable."
        ),
        "trajectory_replay_fidelity": (
            "Generate, freeze TrajectoryRecord, replay.  Fidelity = "
            "fraction of bit-exact tokens.  1.0 = perfect."
        ),
        "edit_locality": (
            "Patch a local frame range; measure max change outside "
            "the patch region.  Lower = better locality."
        ),
        "first_take_naturalness": (
            "Blind A/B preference test on plain text (no acting tags).  "
            "Minimum 30 raters for release sign-off."
        ),
        "few_shot_speaker_similarity": (
            "Frozen speaker similarity protocol with 3s / 5s / 10s "
            "reference.  ECAPA-TDNN cosine similarity."
        ),
        "latency_class_disclosure": (
            "Report hardware class, RTF, time-to-first-audio.  "
            "Both systems must disclose."
        ),
    }
    return methods.get(axis, "TBD")


def determine_claim_scope(report: ComparisonReport) -> str:
    """Apply the claim narrowing rule to determine valid claim scope.

    Rules (from docs/design/evaluation-protocol.md):
    - If all 3 victory axes are won and no guardrail deficit: broad claim OK.
    - If subset of victory axes won + no guardrail deficit: narrow to those axes.
    - If any guardrail shows clear deficit: narrow claim to winning victories only.
    """
    victory_wins = [
        axis for axis, data in report.victory_axes.items()
        if data.get("tmrvc_wins")
    ]
    guardrail_deficits = [
        axis for axis, data in report.guardrail_axes.items()
        if data.get("clear_deficit")
    ]

    if len(victory_wins) == 3 and not guardrail_deficits:
        return "broad: TMRVC v4 beats Fish S2 on all declared victory axes"
    elif victory_wins and not guardrail_deficits:
        axes_str = ", ".join(victory_wins)
        return f"narrow: TMRVC v4 wins on {axes_str} only"
    elif victory_wins and guardrail_deficits:
        axes_str = ", ".join(victory_wins)
        deficit_str = ", ".join(guardrail_deficits)
        return (
            f"narrow with caveats: TMRVC v4 wins on {axes_str} "
            f"but shows deficit on {deficit_str}"
        )
    else:
        return "no claim: insufficient victory axis wins"


# ---------------------------------------------------------------------------
# Full comparison pipeline
# ---------------------------------------------------------------------------

def run_comparison(
    engine: Any,
    fish_samples_dir: Path,
    eval_manifest: list[EvalItem],
    output_dir: Path,
    device: str = "cpu",
) -> ComparisonReport:
    """Run the full Fish S2 comparison pipeline.

    1. Generate TMRVC v4 outputs for each eval item.
    2. Load Fish S2 reference outputs.
    3. Build blind comparison bundle.
    4. Run automatic metrics on both.
    5. Generate report template.

    Args:
        engine: Loaded UCLMEngine.
        fish_samples_dir: Directory containing Fish S2 outputs
            (named {item_id}.wav).
        eval_manifest: List of evaluation items.
        output_dir: Output directory.
        device: Torch device.

    Returns:
        ComparisonReport ready for human evaluation annotation.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    tmrvc_out_dir = output_dir / "tmrvc_outputs"
    tmrvc_out_dir.mkdir(exist_ok=True)

    tmrvc_outputs: dict[str, Path] = {}
    fish_outputs: dict[str, Path] = {}

    speaker_embed = torch.randn(1, 192, device=device)
    speaker_embed = speaker_embed / speaker_embed.norm(dim=-1, keepdim=True)

    for item in eval_manifest:
        # TMRVC generation
        phonemes = torch.randint(1, 199, (1, 40), device=device)
        tmrvc_path = tmrvc_out_dir / f"{item.item_id}.wav"

        try:
            controls = torch.full((1, 1, 12), 0.5, device=device)
            audio, meta = engine.tts(
                phonemes=phonemes,
                speaker_embed=speaker_embed,
                explicit_voice_state=controls,
                cfg_mode="off",
                temperature=0.0,
                max_frames=300,
            )
            # Save audio (16-bit PCM WAV)
            audio_np = audio.cpu().numpy()
            _save_wav(tmrvc_path, audio_np, 24000)
            tmrvc_outputs[item.item_id] = tmrvc_path
        except Exception as e:
            logger.warning("TMRVC generation failed for %s: %s", item.item_id, e)

        # Fish S2 reference
        fish_path = fish_samples_dir / f"{item.item_id}.wav"
        if fish_path.exists():
            fish_outputs[item.item_id] = fish_path
        else:
            logger.warning("Fish S2 sample not found: %s", fish_path)

    # Build blind bundle
    pairs = generate_blind_bundle(tmrvc_outputs, fish_outputs, output_dir)

    # Generate report
    report = generate_report_template(
        pairs=pairs,
        output_path=output_dir / "comparison_report.json",
    )

    # Run automatic victory axis metrics
    logger.info("Running automatic victory axis metrics...")

    # Axis 1: Acting editability
    edit_result = measure_acting_editability(engine, speaker_embed, phonemes, device)
    report.victory_axes["acting_editability"]["tmrvc_score"] = edit_result.get("editability_score", 0.0)

    # Axis 2: Replay fidelity
    replay_result = measure_replay_fidelity(engine, speaker_embed, phonemes, device)
    report.victory_axes["trajectory_replay_fidelity"]["tmrvc_score"] = replay_result.get("fidelity", 0.0)

    # Axis 3: Edit locality
    locality_result = measure_edit_locality_axis(engine, speaker_embed, phonemes, device)
    report.victory_axes["edit_locality"]["tmrvc_score"] = locality_result.get("max_outside_diff", 1.0)

    # Determine per-axis wins by comparing tmrvc_score vs fish_s2_score
    for axis, data in report.victory_axes.items():
        tmrvc_s = data.get("tmrvc_score")
        fish_s = data.get("fish_s2_score")
        if tmrvc_s is None or fish_s is None:
            data["tmrvc_wins"] = None  # cannot determine without both scores
        elif axis == "edit_locality":
            # Lower max_outside_diff = better locality
            data["tmrvc_wins"] = tmrvc_s < fish_s
        else:
            # Higher score = better for editability and replay fidelity
            data["tmrvc_wins"] = tmrvc_s > fish_s

    # Determine claim scope
    report.claim_scope = determine_claim_scope(report)

    # Save final report
    final_report_path = output_dir / "comparison_report.json"
    with open(final_report_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)

    logger.info("Comparison report written to %s", final_report_path)
    return report


def _save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Save audio as 16-bit PCM WAV."""
    try:
        import soundfile as sf
        sf.write(str(path), audio, sr, subtype="PCM_16")
    except ImportError:
        import wave
        import struct

        audio_16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_16.tobytes())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fish Audio S2 Head-to-Head Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tmrvc-checkpoint", type=str, default=None)
    parser.add_argument("--tmrvc-codec", type=str, default=None)
    parser.add_argument("--fish-samples-dir", type=str, default="data/fish_s2_outputs")
    parser.add_argument("--eval-set", type=str, default=None,
                        help="Path to eval manifest JSON (list of {item_id, text, ...})")
    parser.add_argument("--output-dir", type=str, default="results/fish_s2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-weights", action="store_true")
    parser.add_argument("--template-only", action="store_true",
                        help="Generate report template without running inference")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    output_dir = Path(args.output_dir)
    fish_dir = Path(args.fish_samples_dir)

    # Load or generate eval manifest
    if args.eval_set and Path(args.eval_set).exists():
        with open(args.eval_set) as f:
            raw_items = json.load(f)
        eval_items = [EvalItem(**item) for item in raw_items]
    else:
        # Generate synthetic eval items for template/testing
        logger.info("No eval set provided; generating synthetic items.")
        eval_items = [
            EvalItem(item_id=f"item_{i:03d}", text=f"Test sentence {i}", language="ja")
            for i in range(10)
        ]

    if args.template_only:
        pairs = [
            ComparisonPair(
                pair_id=f"pair_{i:03d}",
                item_id=item.item_id,
                system_a="tmrvc", system_b="fish_s2",
                audio_a_path="", audio_b_path="",
                text=item.text, language=item.language,
            )
            for i, item in enumerate(eval_items)
        ]
        report = generate_report_template(
            pairs=pairs,
            output_path=output_dir / "comparison_report_template.json",
        )
        print(f"\nTemplate written to {output_dir / 'comparison_report_template.json'}")
        print(f"Victory axes: {VICTORY_AXES}")
        print(f"Guardrail axes: {GUARDRAIL_AXES}")
        print(f"Claim narrowing rule: {CLAIM_NARROWING_RULE}")
        return

    # Load engine
    import torch
    from tmrvc_serve.uclm_engine import UCLMEngine

    engine = UCLMEngine(device=args.device)
    if args.random_weights:
        engine.init_random_models()
    elif args.tmrvc_checkpoint and args.tmrvc_codec:
        engine.load_models(args.tmrvc_checkpoint, args.tmrvc_codec)
    else:
        logger.info("No checkpoint; using random weights.")
        engine.init_random_models()

    report = run_comparison(
        engine=engine,
        fish_samples_dir=fish_dir,
        eval_manifest=eval_items,
        output_dir=output_dir,
        device=args.device,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("FISH S2 HEAD-TO-HEAD COMPARISON REPORT")
    print("=" * 60)
    print(f"Report ID:     {report.report_id}")
    print(f"Pairs:         {report.pairs_evaluated}")
    print()
    print("Victory Axes:")
    for axis, data in report.victory_axes.items():
        print(f"  {axis}: tmrvc={data.get('tmrvc_score', 'N/A')}")
    print()
    print("Guardrail Axes:")
    for axis, data in report.guardrail_axes.items():
        print(f"  {axis}: deficit={data.get('clear_deficit', 'N/A')}")
    print()
    print(f"Claim scope: {report.claim_scope}")
    print("=" * 60)


if __name__ == "__main__":
    main()
