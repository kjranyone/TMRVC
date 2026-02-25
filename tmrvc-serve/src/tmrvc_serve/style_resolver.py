"""Style resolution: presets, blending, dialogue dynamics, inline stage overlay."""

from __future__ import annotations

import dataclasses
import logging

from tmrvc_core.dialogue_types import DialogueTurn, StyleParams
from tmrvc_serve.schemas import StylePreset

logger = logging.getLogger(__name__)

_HINT_BLEND_WEIGHT = 0.35
_SITUATION_BLEND_WEIGHT = 0.20
_INLINE_STAGE_BLEND_WEIGHT = 0.60
_BACKCHANNEL_TOKENS = frozenset({
    "uh-huh", "yeah", "yes", "no", "okay", "ok",
    "うん", "はい", "ええ", "そう", "そうだね",
    "嗯", "好", "是", "对",
    "응", "네", "그래",
})


@dataclasses.dataclass(frozen=True)
class StylePresetConfig:
    """High-level preset for ASMR-style speaking characteristics."""

    emotion: str | None = None
    delta_valence: float = 0.0
    delta_arousal: float = 0.0
    delta_dominance: float = 0.0
    delta_speech_rate: float = 0.0
    delta_energy: float = 0.0
    delta_pitch_range: float = 0.0
    speed_multiplier: float = 1.0
    sentence_pause_ms: int = 120
    auto_style: bool = True


_STYLE_PRESET_TABLE: dict[str, StylePresetConfig] = {
    "default": StylePresetConfig(),
    "asmr_soft": StylePresetConfig(
        emotion="whisper",
        delta_valence=0.10,
        delta_arousal=-0.35,
        delta_speech_rate=-0.25,
        delta_energy=-0.55,
        delta_pitch_range=-0.10,
        speed_multiplier=0.90,
        sentence_pause_ms=220,
        auto_style=False,
    ),
    "asmr_intimate": StylePresetConfig(
        emotion="whisper",
        delta_valence=0.15,
        delta_arousal=-0.45,
        delta_speech_rate=-0.35,
        delta_energy=-0.65,
        delta_pitch_range=-0.20,
        speed_multiplier=0.82,
        sentence_pause_ms=280,
        auto_style=False,
    ),
}


def _clamp_style(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _clamp_speed(v: float) -> float:
    return max(0.5, min(2.0, v))


def _merge_reasoning(*parts: str | None) -> str:
    chunks = [p.strip() for p in parts if p and p.strip()]
    return "; ".join(chunks)


def _blend_style_values(base: float, overlay: float, overlay_weight: float) -> float:
    return _clamp_style((base * (1.0 - overlay_weight)) + (overlay * overlay_weight))


def _resolve_style_preset(
    base_style: StyleParams | None,
    preset: StylePreset,
) -> tuple[StyleParams | None, StylePresetConfig]:
    """Apply high-level style preset to a base style."""
    cfg = _STYLE_PRESET_TABLE[str(preset)]
    if preset == "default" and base_style is None:
        return None, cfg

    src = base_style or StyleParams.neutral()
    result = StyleParams(
        emotion=cfg.emotion or src.emotion,
        valence=_clamp_style(src.valence + cfg.delta_valence),
        arousal=_clamp_style(src.arousal + cfg.delta_arousal),
        dominance=_clamp_style(src.dominance + cfg.delta_dominance),
        speech_rate=_clamp_style(src.speech_rate + cfg.delta_speech_rate),
        energy=_clamp_style(src.energy + cfg.delta_energy),
        pitch_range=_clamp_style(src.pitch_range + cfg.delta_pitch_range),
        reasoning=(src.reasoning + "; " if src.reasoning else "") + f"preset={preset}",
    )
    return result, cfg


def _resolve_effective_speed(base_speed: float, cfg: StylePresetConfig) -> float:
    return _clamp_speed(base_speed * cfg.speed_multiplier)


def _resolve_sentence_pause(base_pause_ms: int, delta_ms: int) -> int:
    return max(0, min(1600, int(base_pause_ms + delta_ms)))


def _resolve_stage_speed(base_speed: float, stage_scale: float) -> float:
    return _clamp_speed(base_speed * stage_scale)


def _apply_inline_stage_overlay(
    style: StyleParams | None,
    stage_overlay: object | None,
) -> StyleParams | None:
    """Apply inline stage overlay using **additive** blending.

    Unlike hint/situation blends (which interpolate two absolute styles),
    the stage overlay values are *deltas from zero*.  Interpolation would
    incorrectly drag all base parameters toward 0.  Instead we add
    ``delta * weight`` to each base parameter.
    """
    if not isinstance(stage_overlay, StyleParams):
        return style
    if style is None:
        # No base — treat deltas as absolute, scaled by weight.
        w = _INLINE_STAGE_BLEND_WEIGHT
        return StyleParams(
            emotion=stage_overlay.emotion,
            valence=_clamp_style(stage_overlay.valence * w),
            arousal=_clamp_style(stage_overlay.arousal * w),
            dominance=_clamp_style(stage_overlay.dominance * w),
            speech_rate=_clamp_style(stage_overlay.speech_rate * w),
            energy=_clamp_style(stage_overlay.energy * w),
            pitch_range=_clamp_style(stage_overlay.pitch_range * w),
            reasoning=_merge_reasoning(stage_overlay.reasoning, "inline_stage"),
        )

    w = _INLINE_STAGE_BLEND_WEIGHT
    # Inline stage emotion has explicit user intent — override even
    # non-neutral base emotion when the stage direction sets one.
    emotion = style.emotion
    if stage_overlay.emotion != "neutral":
        emotion = stage_overlay.emotion

    return StyleParams(
        emotion=emotion,
        valence=_clamp_style(style.valence + stage_overlay.valence * w),
        arousal=_clamp_style(style.arousal + stage_overlay.arousal * w),
        dominance=_clamp_style(style.dominance + stage_overlay.dominance * w),
        speech_rate=_clamp_style(style.speech_rate + stage_overlay.speech_rate * w),
        energy=_clamp_style(style.energy + stage_overlay.energy * w),
        pitch_range=_clamp_style(style.pitch_range + stage_overlay.pitch_range * w),
        reasoning=_merge_reasoning(style.reasoning, stage_overlay.reasoning, "inline_stage"),
    )


def _blend_styles(
    base: StyleParams | None,
    overlay: StyleParams | None,
    overlay_weight: float,
    reason_tag: str,
) -> StyleParams | None:
    if base is None and overlay is None:
        return None
    if base is None:
        assert overlay is not None
        return StyleParams(
            emotion=overlay.emotion,
            valence=overlay.valence,
            arousal=overlay.arousal,
            dominance=overlay.dominance,
            speech_rate=overlay.speech_rate,
            energy=overlay.energy,
            pitch_range=overlay.pitch_range,
            reasoning=_merge_reasoning(overlay.reasoning, reason_tag),
        )
    if overlay is None:
        return base

    emotion = base.emotion
    if base.emotion == "neutral" and overlay.emotion != "neutral":
        emotion = overlay.emotion

    return StyleParams(
        emotion=emotion,
        valence=_blend_style_values(base.valence, overlay.valence, overlay_weight),
        arousal=_blend_style_values(base.arousal, overlay.arousal, overlay_weight),
        dominance=_blend_style_values(base.dominance, overlay.dominance, overlay_weight),
        speech_rate=_blend_style_values(base.speech_rate, overlay.speech_rate, overlay_weight),
        energy=_blend_style_values(base.energy, overlay.energy, overlay_weight),
        pitch_range=_blend_style_values(base.pitch_range, overlay.pitch_range, overlay_weight),
        reasoning=_merge_reasoning(base.reasoning, overlay.reasoning, reason_tag),
    )


def _is_backchannel(text: str) -> bool:
    normalized = text.strip().lower().strip(" \t\r\n.。!?！？…")
    return normalized in _BACKCHANNEL_TOKENS


def _apply_dialogue_dynamics(
    style: StyleParams | None,
    history: list[DialogueTurn],
    speaker: str | None,
    text: str,
) -> StyleParams | None:
    if style is None or not history:
        return style

    prev_turn = history[-1]
    valence = style.valence
    arousal = style.arousal
    dominance = style.dominance
    speech_rate = style.speech_rate
    energy = style.energy
    pitch_range = style.pitch_range
    tags: list[str] = []

    turn_changed = bool(speaker) and prev_turn.speaker != speaker
    if turn_changed and ("?" in prev_turn.text or "？" in prev_turn.text):
        arousal += 0.12
        energy += 0.08
        pitch_range += 0.10
        tags.append("reply_to_question")

    if turn_changed and prev_turn.emotion in {"angry", "excited"}:
        arousal += 0.08
        dominance += 0.08
        tags.append("carry_tension")

    if turn_changed and prev_turn.emotion in {"sad", "fearful", "whisper"}:
        valence -= 0.08
        speech_rate -= 0.08
        energy -= 0.08
        tags.append("carry_softness")

    if _is_backchannel(text):
        speech_rate -= 0.08
        energy -= 0.10
        pitch_range -= 0.05
        tags.append("backchannel")

    return StyleParams(
        emotion=style.emotion,
        valence=_clamp_style(valence),
        arousal=_clamp_style(arousal),
        dominance=_clamp_style(dominance),
        speech_rate=_clamp_style(speech_rate),
        energy=_clamp_style(energy),
        pitch_range=_clamp_style(pitch_range),
        reasoning=_merge_reasoning(style.reasoning, ",".join(tags) if tags else None),
    )


def _predict_style_rule_based(
    text: str,
    character: object,
    context_predictor: object | None = None,
) -> StyleParams:
    if context_predictor is not None:
        return context_predictor.predict_rule_based(text, character)

    from tmrvc_core.text_utils import infer_sentence_style

    fallback = infer_sentence_style(text, character.language, StyleParams.neutral())
    if isinstance(fallback, StyleParams):
        return fallback
    return StyleParams.neutral()


async def _predict_style_from_inputs(
    *,
    character: object,
    text: str,
    emotion: str | None,
    history: list[DialogueTurn],
    situation: str | None,
    hint: str | None,
    speaker: str | None,
    context_predictor: object | None = None,
) -> StyleParams | None:
    style: StyleParams | None = None

    if emotion:
        style = StyleParams(emotion=emotion, reasoning="emotion_override")
    else:
        if context_predictor is not None:
            try:
                if history or situation:
                    style = await context_predictor.predict(
                        character,
                        history,
                        text,
                        situation,
                    )
                else:
                    style = context_predictor.predict_rule_based(text, character)
            except Exception as e:
                logger.warning("Context prediction failed, using rule-based fallback: %s", e)
                style = None
        if style is None:
            style = _predict_style_rule_based(text, character, context_predictor)

    style = _apply_dialogue_dynamics(style, history, speaker, text)

    if situation:
        situation_style = _predict_style_rule_based(situation, character, context_predictor)
        style = _blend_styles(
            style,
            situation_style,
            overlay_weight=_SITUATION_BLEND_WEIGHT,
            reason_tag="situation_soft",
        )

    if hint:
        hint_style = _predict_style_rule_based(hint, character, context_predictor)
        style = _blend_styles(
            style,
            hint_style,
            overlay_weight=_HINT_BLEND_WEIGHT,
            reason_tag="hint_soft",
        )

    return style
