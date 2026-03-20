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
_BACKCHANNEL_TOKENS = frozenset(
    {
        "uh-huh",
        "yeah",
        "yes",
        "no",
        "okay",
        "ok",
        "うん",
        "はい",
        "ええ",
        "そう",
        "そうだね",
        "嗯",
        "好",
        "是",
        "对",
        "응",
        "네",
        "그래",
    }
)


@dataclasses.dataclass(frozen=True)
class StylePresetConfig:
    """High-level preset for speaking characteristics."""

    emotion: str | None = None
    delta_breathiness: float = 0.0
    delta_tension: float = 0.0
    delta_arousal: float = 0.0
    delta_valence: float = 0.0
    delta_roughness: float = 0.0
    delta_voicing: float = 0.0
    delta_energy: float = 0.0
    speech_rate_multiplier: float = 1.0
    sentence_pause_ms: int = 120
    auto_style: bool = True


_STYLE_PRESET_TABLE: dict[str, StylePresetConfig] = {
    "default": StylePresetConfig(),
    "asmr_soft": StylePresetConfig(
        emotion="whisper",
        delta_breathiness=0.6,
        delta_tension=-0.3,
        delta_arousal=-0.4,
        delta_voicing=-0.5,
        delta_energy=-0.4,
        speech_rate_multiplier=0.85,
        sentence_pause_ms=250,
        auto_style=False,
    ),
    "asmr_intimate": StylePresetConfig(
        emotion="whisper",
        delta_breathiness=0.8,
        delta_tension=-0.4,
        delta_arousal=-0.5,
        delta_voicing=-0.7,
        delta_energy=-0.5,
        speech_rate_multiplier=0.75,
        sentence_pause_ms=300,
        auto_style=False,
    ),
    "angry_intense": StylePresetConfig(
        emotion="angry",
        delta_tension=0.7,
        delta_arousal=0.6,
        delta_roughness=0.3,
        delta_energy=0.5,
        speech_rate_multiplier=1.2,
        sentence_pause_ms=80,
    ),
}


def _clamp_unit(v: float) -> float:
    return max(0.0, min(1.0, v))


def _clamp_valence(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _clamp_speed(v: float) -> float:
    return max(0.5, min(2.0, v))


def _merge_reasoning(*parts: str | None) -> str:
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        s = str(p).strip()
        if s:
            out.append(s)
    return "; ".join(out)


def _blend_style_values(base: float, overlay: float, overlay_weight: float) -> float:
    w = max(0.0, min(1.0, overlay_weight))
    return (1.0 - w) * base + w * overlay


def _resolve_style_preset(
    base_style: StyleParams | None,
    preset: StylePreset,
) -> tuple[StyleParams | None, StylePresetConfig]:
    """Apply high-level style preset to a base style."""
    cfg = _STYLE_PRESET_TABLE.get(str(preset), _STYLE_PRESET_TABLE["default"])
    if preset == "default" and base_style is None:
        return None, cfg

    src = base_style or StyleParams.neutral()
    result = StyleParams(
        emotion=cfg.emotion or src.emotion,
        pitch_level=src.pitch_level,
        pitch_range=src.pitch_range,
        energy=_clamp_unit(src.energy + cfg.delta_energy),
        tension=_clamp_unit(src.tension + cfg.delta_tension),
        spectral_tilt=src.spectral_tilt,
        breathiness=_clamp_unit(src.breathiness + cfg.delta_breathiness),
        roughness=_clamp_unit(src.roughness + cfg.delta_roughness),
        openness=src.openness,
        aperiodicity=src.aperiodicity,
        formant_shift=src.formant_shift,
        vocal_effort=src.vocal_effort,
        creak=src.creak,
        arousal=_clamp_unit(src.arousal + cfg.delta_arousal),
        valence=_clamp_valence(src.valence + cfg.delta_valence),
        voicing=_clamp_unit(src.voicing + cfg.delta_voicing),
        speech_rate=_clamp_speed(src.speech_rate * cfg.speech_rate_multiplier),
        reasoning=_merge_reasoning(
            src.reasoning, f"preset={preset}" if preset != "default" else None
        ),
    )
    return result, cfg


def _resolve_effective_speed(base_speed: float, cfg: StylePresetConfig) -> float:
    return _clamp_speed(base_speed * cfg.speech_rate_multiplier)


def _resolve_sentence_pause(base_pause_ms: int, delta_ms: int) -> int:
    return max(0, min(1600, int(base_pause_ms + delta_ms)))


def _resolve_stage_speed(base_speed: float, stage_scale: float) -> float:
    return _clamp_speed(base_speed * stage_scale)


def _apply_inline_stage_overlay(
    style: StyleParams | None,
    stage_overlay: object | None,
) -> StyleParams | None:
    """Apply inline stage overlay as additive deltas from neutral."""
    if not isinstance(stage_overlay, StyleParams):
        return style

    base = style or StyleParams.neutral()
    neutral = StyleParams.neutral()
    w = _INLINE_STAGE_BLEND_WEIGHT

    def d(name: str) -> float:
        return float(getattr(stage_overlay, name) - getattr(neutral, name))

    emotion = base.emotion
    if stage_overlay.emotion != neutral.emotion:
        emotion = stage_overlay.emotion

    return StyleParams(
        emotion=emotion,
        pitch_level=_clamp_unit(base.pitch_level + d("pitch_level") * w),
        pitch_range=_clamp_unit(base.pitch_range + d("pitch_range") * w),
        energy=_clamp_unit(base.energy + d("energy") * w),
        tension=_clamp_unit(base.tension + d("tension") * w),
        spectral_tilt=_clamp_unit(base.spectral_tilt + d("spectral_tilt") * w),
        breathiness=_clamp_unit(base.breathiness + d("breathiness") * w),
        roughness=_clamp_unit(base.roughness + d("roughness") * w),
        openness=_clamp_unit(base.openness + d("openness") * w),
        aperiodicity=_clamp_unit(base.aperiodicity + d("aperiodicity") * w),
        formant_shift=_clamp_unit(base.formant_shift + d("formant_shift") * w),
        vocal_effort=_clamp_unit(base.vocal_effort + d("vocal_effort") * w),
        creak=_clamp_unit(base.creak + d("creak") * w),
        arousal=_clamp_unit(base.arousal + d("arousal") * w),
        valence=_clamp_valence(base.valence + d("valence") * w),
        voicing=_clamp_unit(base.voicing + d("voicing") * w),
        speech_rate=_clamp_speed(base.speech_rate + d("speech_rate") * w),
        reasoning=_merge_reasoning(base.reasoning, stage_overlay.reasoning, "inline_stage"),
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
            pitch_level=overlay.pitch_level,
            pitch_range=overlay.pitch_range,
            energy=overlay.energy,
            tension=overlay.tension,
            spectral_tilt=overlay.spectral_tilt,
            breathiness=overlay.breathiness,
            roughness=overlay.roughness,
            openness=overlay.openness,
            aperiodicity=overlay.aperiodicity,
            formant_shift=overlay.formant_shift,
            vocal_effort=overlay.vocal_effort,
            creak=overlay.creak,
            arousal=overlay.arousal,
            valence=overlay.valence,
            voicing=overlay.voicing,
            speech_rate=overlay.speech_rate,
            reasoning=_merge_reasoning(overlay.reasoning, reason_tag),
        )
    if overlay is None:
        return base

    emotion = base.emotion
    if base.emotion == "neutral" and overlay.emotion != "neutral":
        emotion = overlay.emotion

    return StyleParams(
        emotion=emotion,
        pitch_level=_clamp_unit(
            _blend_style_values(base.pitch_level, overlay.pitch_level, overlay_weight)
        ),
        pitch_range=_clamp_unit(
            _blend_style_values(base.pitch_range, overlay.pitch_range, overlay_weight)
        ),
        energy=_clamp_unit(
            _blend_style_values(base.energy, overlay.energy, overlay_weight)
        ),
        tension=_clamp_unit(
            _blend_style_values(base.tension, overlay.tension, overlay_weight)
        ),
        spectral_tilt=_clamp_unit(
            _blend_style_values(base.spectral_tilt, overlay.spectral_tilt, overlay_weight)
        ),
        breathiness=_clamp_unit(
            _blend_style_values(base.breathiness, overlay.breathiness, overlay_weight)
        ),
        roughness=_clamp_unit(
            _blend_style_values(base.roughness, overlay.roughness, overlay_weight)
        ),
        openness=_clamp_unit(
            _blend_style_values(base.openness, overlay.openness, overlay_weight)
        ),
        aperiodicity=_clamp_unit(
            _blend_style_values(base.aperiodicity, overlay.aperiodicity, overlay_weight)
        ),
        formant_shift=_clamp_unit(
            _blend_style_values(base.formant_shift, overlay.formant_shift, overlay_weight)
        ),
        vocal_effort=_clamp_unit(
            _blend_style_values(base.vocal_effort, overlay.vocal_effort, overlay_weight)
        ),
        creak=_clamp_unit(
            _blend_style_values(base.creak, overlay.creak, overlay_weight)
        ),
        arousal=_clamp_unit(
            _blend_style_values(base.arousal, overlay.arousal, overlay_weight)
        ),
        valence=_clamp_valence(
            _blend_style_values(base.valence, overlay.valence, overlay_weight)
        ),
        voicing=_clamp_unit(
            _blend_style_values(base.voicing, overlay.voicing, overlay_weight)
        ),
        speech_rate=_clamp_speed(
            _blend_style_values(base.speech_rate, overlay.speech_rate, overlay_weight)
        ),
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
    tension = style.tension
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
        tension += 0.10
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
        pitch_level=style.pitch_level,
        pitch_range=pitch_range,
        energy=_clamp_unit(energy),
        tension=_clamp_unit(tension),
        spectral_tilt=style.spectral_tilt,
        breathiness=style.breathiness,
        roughness=style.roughness,
        openness=style.openness,
        aperiodicity=style.aperiodicity,
        formant_shift=style.formant_shift,
        vocal_effort=style.vocal_effort,
        creak=style.creak,
        arousal=_clamp_unit(arousal),
        valence=_clamp_valence(valence),
        voicing=style.voicing,
        speech_rate=_clamp_speed(speech_rate),
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
