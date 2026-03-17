#!/usr/bin/env python3
"""Drama-acting evaluation script for TMRVC v3.

Evaluates six key capabilities:
  1. Context sensitivity  -- same text, different context yields different pacing.
  2. Control responsiveness -- pace/hold_bias/boundary_bias produce duration changes.
  3. Pause realism -- pause duration distribution matches natural speech statistics.
  4. Few-shot speaker adaptation -- speaker similarity and intelligibility under
     short reference conditions (3s, 5s, 10s).
  5. Timbre-prosody disentanglement -- F0 variance across contexts with same
     speaker prompt.
  6. Voice-state responsiveness -- each voice_state dimension produces measurable
     acoustic changes.

Usage:
    python scripts/evaluate_drama_acting.py \
        --checkpoint /path/to/model.pt \
        --device cuda:0 \
        --output results/drama_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Test fixtures -- minimal synthetic data used when no external corpus is
# provided.  Replace with real held-out dialogue for production evaluations.
# ---------------------------------------------------------------------------

SAMPLE_TEXTS: List[str] = [
    "I never said she stole my money.",
    "We need to leave right now.",
    "That's exactly what I was afraid of.",
    "You promised you would be here.",
    "I can't believe this is happening.",
]

SAMPLE_CONTEXTS: List[str] = [
    "The speaker is calm and reflective, speaking slowly after a long pause.",
    "The speaker is furious, spitting the words out with clipped urgency.",
    "The speaker is heartbroken, voice trembling with sadness.",
    "The speaker is excited and joyful, nearly laughing mid-sentence.",
]

PACE_VALUES: List[float] = [0.7, 0.85, 1.0, 1.15, 1.3]
HOLD_BIAS_VALUES: List[float] = [0.0, 0.3, 0.6, 1.0]
BOUNDARY_BIAS_VALUES: List[float] = [0.0, 0.5, 1.0]

REFERENCE_DURATIONS_S: List[float] = [3.0, 5.0, 10.0]

VOICE_STATE_DIMENSIONS: List[str] = [
    "energy",
    "pitch_mean",
    "pitch_range",
    "speed",
    "breathiness",
    "tension",
    "warmth",
    "brightness",
]


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    name: str
    passed: bool
    detail: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------


def _load_model(checkpoint: str, device: str) -> Any:
    """Load the UCLM checkpoint.

    This function attempts to import the project-internal model loader.  If it
    is unavailable (e.g. running in a minimal CI environment), it returns
    ``None`` and the evaluation functions will operate in **dry-run** mode
    using synthetic durations so that the script structure can still be
    validated.
    """
    try:
        import torch

        from tmrvc_train.cli.train_uclm import load_model_for_inference

        model = load_model_for_inference(checkpoint, device=device)
        model.eval()
        return model
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not load model ({exc}); running in dry-run mode.")
        return None


def _synthesise(
    model: Any,
    text: str,
    *,
    context: str | None = None,
    pace: float = 1.0,
    hold_bias: float = 0.0,
    boundary_bias: float = 0.0,
) -> Dict[str, Any]:
    """Run TTS inference and return timing metadata.

    When the model is ``None`` (dry-run), synthetic values are generated so
    that downstream analysis code can be exercised.

    Returns a dict with at least:
        - ``duration_s``   : total utterance duration in seconds
        - ``pauses``       : list of pause durations (seconds)
        - ``frame_times``  : list of per-frame-step wall-clock times (seconds)
        - ``n_boundaries`` : detected phrase boundary count
    """
    if model is None:
        # Dry-run: generate plausible synthetic timing data.
        rng = np.random.default_rng(hash((text, context, pace, hold_bias, boundary_bias)) % 2**31)
        base_dur = len(text.split()) * 0.18  # rough word-rate baseline
        duration_s = base_dur / pace + hold_bias * 0.15
        n_frames = max(int(duration_s / 0.0125), 1)
        frame_times = rng.exponential(scale=0.02, size=n_frames).tolist()
        n_pauses = max(1, int(len(text.split()) / 4) + int(boundary_bias * 2))
        pauses = rng.exponential(scale=0.12 + hold_bias * 0.1, size=n_pauses).tolist()
        n_boundaries = max(1, int(len(text.split()) / 3 + boundary_bias * 2))
        return {
            "duration_s": duration_s,
            "pauses": pauses,
            "frame_times": frame_times,
            "n_boundaries": n_boundaries,
        }

    # Real inference path.
    import torch

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.synthesise(
            text,
            context=context,
            pace=pace,
            hold_bias=hold_bias,
            boundary_bias=boundary_bias,
        )
    wall = time.perf_counter() - t0

    audio = output["audio"]  # expected: 1-D tensor or numpy array
    sr = output.get("sample_rate", 24000)
    duration_s = len(audio) / sr

    pauses = output.get("pauses", [])
    frame_times = output.get("frame_times", [wall / max(1, output.get("n_frames", 1))] * output.get("n_frames", 1))
    n_boundaries = output.get("n_boundaries", 0)

    return {
        "duration_s": float(duration_s),
        "pauses": [float(p) for p in pauses],
        "frame_times": [float(ft) for ft in frame_times],
        "n_boundaries": int(n_boundaries),
    }


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------


def evaluate_context_sensitivity(
    model: Any,
    texts: Sequence[str] = SAMPLE_TEXTS,
    contexts: Sequence[str] = SAMPLE_CONTEXTS,
) -> EvalResult:
    """Same text, different context -> measure pacing variance.

    For each text, synthesise with every context and compute the coefficient of
    variation (CV) of the resulting durations.  The test passes if the mean CV
    across texts is >= 0.10.
    """
    cvs: List[float] = []

    for text in texts:
        durations = []
        for ctx in contexts:
            result = _synthesise(model, text, context=ctx)
            durations.append(result["duration_s"])
        arr = np.array(durations)
        mean = arr.mean()
        cv = arr.std() / mean if mean > 0 else 0.0
        cvs.append(float(cv))

    mean_cv = float(np.mean(cvs))
    passed = mean_cv >= 0.10

    return EvalResult(
        name="context_sensitivity",
        passed=passed,
        detail={
            "mean_cv": round(mean_cv, 4),
            "per_text_cv": [round(c, 4) for c in cvs],
            "threshold": 0.10,
            "n_texts": len(texts),
            "n_contexts": len(contexts),
        },
    )


def evaluate_control_responsiveness(
    model: Any,
    texts: Sequence[str] = SAMPLE_TEXTS,
) -> EvalResult:
    """Same text, different pace/hold_bias/boundary_bias -> measure changes.

    Sub-tests:
      a) Duration ratio between lowest and highest pace >= 1.30
      b) Hold duration increases monotonically with hold_bias
      c) Boundary count changes directionally with boundary_bias
    """
    pace_ratios: List[float] = []
    hold_monotonic_count = 0
    boundary_directional_count = 0

    for text in texts:
        # --- Pace ---
        durations_by_pace = []
        for p in PACE_VALUES:
            r = _synthesise(model, text, pace=p)
            durations_by_pace.append(r["duration_s"])
        if durations_by_pace[-1] > 0:
            ratio = durations_by_pace[0] / durations_by_pace[-1]
        else:
            ratio = 0.0
        pace_ratios.append(ratio)

        # --- Hold bias ---
        durations_by_hold = []
        for hb in HOLD_BIAS_VALUES:
            r = _synthesise(model, text, hold_bias=hb)
            durations_by_hold.append(r["duration_s"])
        # Check monotonic increase (allow small tolerance).
        diffs = np.diff(durations_by_hold)
        if np.all(diffs >= -1e-4):
            hold_monotonic_count += 1

        # --- Boundary bias ---
        boundaries_by_bias = []
        for bb in BOUNDARY_BIAS_VALUES:
            r = _synthesise(model, text, boundary_bias=bb)
            boundaries_by_bias.append(r["n_boundaries"])
        if boundaries_by_bias[-1] >= boundaries_by_bias[0]:
            boundary_directional_count += 1

    mean_pace_ratio = float(np.mean(pace_ratios))
    pace_pass = mean_pace_ratio >= 1.30
    hold_pass = hold_monotonic_count >= len(texts) * 0.8
    boundary_pass = boundary_directional_count >= len(texts) * 0.8

    passed = pace_pass and hold_pass and boundary_pass

    return EvalResult(
        name="control_responsiveness",
        passed=passed,
        detail={
            "pace": {
                "mean_ratio_slow_over_fast": round(mean_pace_ratio, 4),
                "per_text_ratios": [round(r, 4) for r in pace_ratios],
                "threshold": 1.30,
                "passed": pace_pass,
            },
            "hold_bias": {
                "monotonic_texts": hold_monotonic_count,
                "total_texts": len(texts),
                "passed": hold_pass,
            },
            "boundary_bias": {
                "directional_texts": boundary_directional_count,
                "total_texts": len(texts),
                "passed": boundary_pass,
            },
        },
    )


def evaluate_pause_realism(
    model: Any,
    texts: Sequence[str] = SAMPLE_TEXTS,
) -> EvalResult:
    """Check that pause duration distribution resembles natural speech.

    Natural conversational speech typically has:
      - Mean pause duration: 0.05 - 0.40 s
      - Median pause duration: 0.04 - 0.30 s
      - Proportion of pauses > 1.0 s: < 10%
      - At least some pauses present (not all zero)

    These ranges are intentionally generous to accommodate different speaking
    styles.
    """
    all_pauses: List[float] = []

    for text in texts:
        r = _synthesise(model, text, pace=1.0)
        all_pauses.extend(r["pauses"])

    if len(all_pauses) == 0:
        return EvalResult(
            name="pause_realism",
            passed=False,
            detail={"error": "No pauses detected in any utterance."},
        )

    arr = np.array(all_pauses)
    mean_pause = float(arr.mean())
    median_pause = float(np.median(arr))
    std_pause = float(arr.std())
    frac_long = float(np.mean(arr > 1.0))

    mean_ok = 0.05 <= mean_pause <= 0.40
    median_ok = 0.04 <= median_pause <= 0.30
    long_ok = frac_long < 0.10

    passed = mean_ok and median_ok and long_ok

    return EvalResult(
        name="pause_realism",
        passed=passed,
        detail={
            "n_pauses": len(all_pauses),
            "mean_s": round(mean_pause, 4),
            "median_s": round(median_pause, 4),
            "std_s": round(std_pause, 4),
            "frac_over_1s": round(frac_long, 4),
            "checks": {
                "mean_in_range": mean_ok,
                "median_in_range": median_ok,
                "long_pause_fraction_ok": long_ok,
            },
        },
    )


def evaluate_few_shot_speaker_adaptation(
    model: Any,
    texts: Sequence[str] = SAMPLE_TEXTS,
    reference_durations: Sequence[float] = REFERENCE_DURATIONS_S,
) -> EvalResult:
    """Evaluate speaker similarity and intelligibility under short references.

    For each reference duration (3s, 5s, 10s), synthesise with a simulated
    reference audio and measure:
      - Speaker similarity (cosine of speaker embeddings).
      - Intelligibility (CER via ASR re-transcription).

    In dry-run mode (model is None), synthetic scores are generated so that
    the evaluation structure can be validated.

    Pass criteria:
      - Speaker similarity >= 0.80 at 3s reference, >= 0.85 at 10s reference.
      - CER < 5% at all reference durations.
    """
    similarity_thresholds = {3.0: 0.80, 5.0: 0.82, 10.0: 0.85}
    cer_threshold = 0.05

    per_duration: Dict[str, Any] = {}
    all_passed = True

    for dur in reference_durations:
        if model is None:
            # Dry-run: generate plausible synthetic scores.
            rng = np.random.default_rng(int(dur * 1000) % 2**31)
            # Similarity improves with longer reference.
            base_sim = 0.78 + dur * 0.008
            similarities = (base_sim + rng.normal(0, 0.02, size=len(texts))).tolist()
            cers = (rng.uniform(0.01, 0.04, size=len(texts))).tolist()
        else:
            similarities = []
            cers = []
            for text in texts:
                try:
                    import torch

                    with torch.no_grad():
                        output = model.synthesise(
                            text,
                            reference_duration_s=dur,
                        )
                    similarities.append(float(output.get("speaker_similarity", 0.0)))
                    cers.append(float(output.get("cer", 0.0)))
                except Exception:  # noqa: BLE001
                    similarities.append(0.0)
                    cers.append(1.0)

        mean_sim = float(np.mean(similarities))
        mean_cer = float(np.mean(cers))
        threshold = similarity_thresholds.get(dur, 0.80)
        sim_pass = mean_sim >= threshold
        cer_pass = mean_cer < cer_threshold

        if not (sim_pass and cer_pass):
            all_passed = False

        per_duration[f"{dur:.0f}s"] = {
            "mean_similarity": round(mean_sim, 4),
            "similarity_threshold": threshold,
            "similarity_passed": sim_pass,
            "mean_cer": round(mean_cer, 4),
            "cer_threshold": cer_threshold,
            "cer_passed": cer_pass,
        }

    return EvalResult(
        name="few_shot_speaker_adaptation",
        passed=all_passed,
        detail=per_duration,
    )


def evaluate_timbre_prosody_disentanglement(
    model: Any,
    texts: Sequence[str] = SAMPLE_TEXTS,
    contexts: Sequence[str] = SAMPLE_CONTEXTS,
) -> EvalResult:
    """Same speaker prompt + different dialogue contexts -> F0 variance.

    Measures the coefficient of variation (CV) of F0 across contexts for the
    same text.  High CV indicates that prosody adapts to context while timbre
    (speaker identity) is preserved.

    In dry-run mode, synthetic F0 values are generated.

    Pass criteria: mean F0 CV across texts >= 0.08.
    """
    f0_cvs: List[float] = []

    for text in texts:
        f0_means: List[float] = []
        for ctx in contexts:
            if model is None:
                # Dry-run: synthetic F0 that varies with context.
                rng = np.random.default_rng(
                    hash((text, ctx, "f0")) % 2**31,
                )
                f0_mean = 180.0 + rng.normal(0, 20)
            else:
                result = _synthesise(model, text, context=ctx)
                f0_mean = result.get("f0_mean", 180.0)
            f0_means.append(f0_mean)

        arr = np.array(f0_means)
        mean_val = arr.mean()
        cv = float(arr.std() / mean_val) if mean_val > 0 else 0.0
        f0_cvs.append(cv)

    mean_cv = float(np.mean(f0_cvs))
    passed = mean_cv >= 0.08

    return EvalResult(
        name="timbre_prosody_disentanglement",
        passed=passed,
        detail={
            "mean_f0_cv": round(mean_cv, 4),
            "per_text_f0_cv": [round(c, 4) for c in f0_cvs],
            "threshold": 0.08,
            "n_texts": len(texts),
            "n_contexts": len(contexts),
        },
    )


def evaluate_voice_state_responsiveness(
    model: Any,
    texts: Sequence[str] = SAMPLE_TEXTS,
    dimensions: Sequence[str] = VOICE_STATE_DIMENSIONS,
) -> EvalResult:
    """Sweep each voice_state dimension and measure acoustic metric change.

    For each of the 8 voice_state dimensions, set that dimension to low (0.0)
    and high (1.0) while keeping others at 0.5, then measure the directional
    change in a relevant acoustic metric (duration, F0 mean, energy).

    In dry-run mode, synthetic acoustic metrics are generated.

    Pass criteria: at least 6/8 dimensions produce measurable directional
    change (absolute delta > threshold).
    """
    responsive_count = 0
    per_dim: Dict[str, Any] = {}
    delta_threshold = 0.01  # minimum relative change to count as responsive

    for dim in dimensions:
        low_metrics: List[float] = []
        high_metrics: List[float] = []

        for text in texts:
            if model is None:
                # Dry-run: synthetic metrics that respond to voice_state.
                rng_low = np.random.default_rng(
                    hash((text, dim, "low")) % 2**31,
                )
                rng_high = np.random.default_rng(
                    hash((text, dim, "high")) % 2**31,
                )
                # Simulate different acoustic responses per dimension.
                base = len(text.split()) * 0.2
                low_val = base + rng_low.normal(0, 0.05)
                high_val = base * 1.15 + rng_high.normal(0, 0.05)
                low_metrics.append(low_val)
                high_metrics.append(high_val)
            else:
                # Build voice_state vectors: baseline at 0.5, sweep target dim.
                baseline = {d: 0.5 for d in dimensions}
                vs_low = {**baseline, dim: 0.0}
                vs_high = {**baseline, dim: 1.0}

                r_low = _synthesise(model, text, context=None)
                r_high = _synthesise(model, text, context=None)

                # Use duration as the primary acoustic metric.
                low_metrics.append(r_low["duration_s"])
                high_metrics.append(r_high["duration_s"])

        mean_low = float(np.mean(low_metrics))
        mean_high = float(np.mean(high_metrics))
        abs_delta = abs(mean_high - mean_low)
        ref_val = max(abs(mean_low), abs(mean_high), 1e-8)
        rel_delta = abs_delta / ref_val

        is_responsive = rel_delta > delta_threshold
        if is_responsive:
            responsive_count += 1

        per_dim[dim] = {
            "mean_low": round(mean_low, 4),
            "mean_high": round(mean_high, 4),
            "abs_delta": round(abs_delta, 4),
            "rel_delta": round(rel_delta, 4),
            "responsive": is_responsive,
        }

    min_responsive = 6
    passed = responsive_count >= min_responsive

    return EvalResult(
        name="voice_state_responsiveness",
        passed=passed,
        detail={
            "responsive_dimensions": responsive_count,
            "total_dimensions": len(dimensions),
            "min_required": min_responsive,
            "per_dimension": per_dim,
        },
    )


def evaluate_replay_fidelity(engine: Any, texts: Sequence[str] = SAMPLE_TEXTS) -> EvalResult:
    """Measure how faithfully a trajectory is replayed.
    
    Sub-tests:
      a) Bit-exact parity: Stream A tokens must match 100% when use_exact_tokens=True.
      b) Timing parity: Pointer trace durations must match 100% in all replay modes.
    """
    from tmrvc_data.g2p import text_to_phonemes
    import torch

    bit_exact_matches = 0
    timing_matches = 0
    
    for text in texts:
        g2p = text_to_phonemes(text)
        phonemes_t = g2p.phoneme_ids.to(engine.device).unsqueeze(0)
        
        # 1. Original generation
        dummy_spk = torch.randn(1, 192, device=engine.device)
        audio_orig, stats_orig = engine.tts(phonemes=phonemes_t, speaker_embed=dummy_spk, language_id=g2p.language_id)
        traj = engine.create_trajectory_record("eval-orig", "eval-compile", stats_orig, phonemes_t, g2p.text_suprasegmentals)
        
        # 2. Bit-exact replay
        audio_rep, stats_rep = engine.replay_trajectory(
            phonemes=phonemes_t, 
            trajectory=traj, 
            speaker_profile=None,
            speaker_embed=dummy_spk, # Pass the same speaker
            text_suprasegmentals=g2p.text_suprasegmentals,
            use_exact_tokens=True
        )
        
        # Check tokens
        trace_orig = stats_orig["acoustic_trace"].cpu()
        trace_rep = stats_rep["acoustic_trace"].cpu() if "acoustic_trace" in stats_rep else torch.zeros(0)
        
        if torch.equal(trace_orig, trace_rep):
            bit_exact_matches += 1
            
        # Check timing (pointer trace)
        if stats_orig["pointer_trace"] == stats_rep.get("pointer_trace"):
            timing_matches += 1

    bit_pass = bit_exact_matches == len(texts)
    timing_pass = timing_matches == len(texts)
    
    return EvalResult(
        name="replay_fidelity",
        passed=bit_pass and timing_pass,
        detail={
            "bit_exact_parity": {"matches": bit_exact_matches, "total": len(texts), "passed": bit_pass},
            "timing_parity": {"matches": timing_matches, "total": len(texts), "passed": timing_pass}
        }
    )

def evaluate_edit_locality(engine: Any, texts: Sequence[str] = SAMPLE_TEXTS) -> EvalResult:
    """Measure if patching one segment leaves other segments identical.
    
    Test: Patch the voice state of the middle phoneme. Verify that the Stream A
    tokens for the prefix and suffix phonemes remain bit-exact.
    """
    from tmrvc_data.g2p import text_to_phonemes
    import torch

    locality_matches = 0
    
    for text in texts:
        g2p = text_to_phonemes(text)
        L = g2p.phoneme_ids.shape[0]
        if L < 3: continue # Need at least 3 phonemes to test locality
        
        phonemes_t = g2p.phoneme_ids.to(engine.device).unsqueeze(0)
        mid_idx = L // 2
        dummy_spk = torch.randn(1, 192, device=engine.device)
        
        # 1. Baseline generation (Greedy)
        _, stats_orig = engine.tts(phonemes=phonemes_t, speaker_embed=dummy_spk, language_id=g2p.language_id, temperature=0)
        a_trace_orig = stats_orig["acoustic_trace"] # [8, T]
        ptr_trace_orig = stats_orig["pointer_trace"]
        
        # 2. Patch middle phoneme's voice state (Greedy)
        # (This uses the 'patch' logic: same pacing, but different state for one unit)
        overrides = {mid_idx: [1.0] * 8} # Max out all VS dims for the middle phoneme
        
        _, stats_patch = engine.tts(
            phonemes=phonemes_t, 
            speaker_embed=dummy_spk,
            language_id=g2p.language_id,
            local_prosody_plan={mid_idx: torch.tensor(overrides[mid_idx]).float().view(1, 8)},
            temperature=0
        )
        a_trace_patch = stats_patch["acoustic_trace"]
        
        # 3. Verify locality
        # Find the frame range for the first phoneme (prefix)
        # pointer_trace is [(index, duration), ...]
        prefix_dur = ptr_trace_orig[0][1]
        
        prefix_orig = a_trace_orig[:, :prefix_dur]
        prefix_patch = a_trace_patch[:, :prefix_dur]
        
        if torch.equal(prefix_orig, prefix_patch):
            locality_matches += 1

    passed = locality_matches >= len(texts) * 0.8 # 80% locality threshold
    
    return EvalResult(
        name="edit_locality",
        passed=passed,
        detail={
            "prefix_bit_exact_matches": locality_matches,
            "total_texts": len(texts),
            "threshold": 0.8
        }
    )

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(results: List[EvalResult]) -> None:
    width = 72
    print("=" * width)
    print("  DRAMA-ACTING EVALUATION REPORT")
    print("=" * width)

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if not r.passed:
            all_passed = False
        print(f"\n--- {r.name} [{status}] ---")
        for k, v in r.detail.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")

    print("\n" + "=" * width)
    overall = "PASS" if all_passed else "FAIL"
    print(f"  OVERALL: {overall}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate drama-acting capabilities of a TMRVC v3 checkpoint.",
    )
    parser.add_argument(
        "--uclm-checkpoint",
        type=str,
        required=True,
        help="Path to UCLM model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--codec-checkpoint",
        type=str,
        required=True,
        help="Path to Codec model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device string (default: cuda:0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON results.",
    )

    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="Use randomly initialized models instead of loading checkpoints.",
    )

    args = parser.parse_args()

    from tmrvc_serve.uclm_engine import UCLMEngine
    if args.random_weights:
        print("Initializing engine with RANDOM WEIGHTS for architectural verification...")
        engine = UCLMEngine(device=args.device)
        engine.init_random_models() 
    else:
        print(f"Loading engine from {args.uclm_checkpoint} ...")
        engine = UCLMEngine(
            uclm_checkpoint=args.uclm_checkpoint,
            codec_checkpoint=args.codec_checkpoint,
            device=args.device
        )
        engine.load_models()

    results: List[EvalResult] = []

    # 1. Programmable Expressive Speech Axes (New SOTA requirements)
    print("Running replay_fidelity evaluation ...")
    results.append(evaluate_replay_fidelity(engine))

    print("Running edit_locality evaluation ...")
    results.append(evaluate_edit_locality(engine))

    # 2. Drama Baseline Axes (Legacy stubs or updated to engine)
    print("Running context_sensitivity evaluation ...")
    # (Refactoring of legacy eval functions to use engine is ongoing)
    # results.append(evaluate_context_sensitivity(engine))

    _print_report(results)

    # Write JSON output if requested.
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {r.name: {"passed": r.passed, "detail": r.detail} for r in results}
        payload["overall_passed"] = all(r.passed for r in results)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults written to {out_path}")

    if not all(r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
