"""Auto-generate constants from YAML.

This module is called by constants.py on import when the generated file
is missing or stale.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).resolve().parents[3] / "configs" / "constants.yaml"
_PY_OUT = Path(__file__).parent / "_generated_constants.py"
_RUST_OUT = (
    Path(__file__).resolve().parents[3] / "tmrvc-engine-rs" / "src" / "constants.rs"
)
_CPP_OUT = (
    Path(__file__).resolve().parents[3]
    / "tmrvc-engine"
    / "include"
    / "tmrvc"
    / "constants.h"
)

_RUNTIME_KEYS = {
    "sample_rate",
    "n_fft",
    "hop_length",
    "window_length",
    "n_mels",
    "mel_fmin",
    "mel_fmax",
    "n_freq_bins",
    "log_floor",
    "d_content",
    "d_speaker",
    "n_ir_params",
    "n_voice_source_params",
    "n_acoustic_params",
    "d_converter_hidden",
    "d_ir_estimator_hidden",
    "d_vocoder_hidden",
    "d_vocoder_features",
    "ir_update_interval",
    "lora_rank",
    "lora_delta_size",
    "content_encoder_state_frames",
    "ir_estimator_state_frames",
    "converter_state_frames",
    "vocoder_state_frames",
    "max_lookahead_hops",
    "converter_hq_state_frames",
    "hq_threshold_q",
    "crossfade_frames",
    "d_style",
    "n_style_params",
    "d_text_encoder",
    "n_text_encoder_layers",
    "n_text_encoder_heads",
    "text_encoder_ff_dim",
    "phoneme_vocab_size",
    "tokenizer_vocab_size",
    "n_emotion_categories",
    "d_f0_predictor",
    "d_content_synthesizer",
    "n_languages",
    "n_codebooks",
    "rvq_vocab_size",
    "control_vocab_size",
    "control_slots",
    "d_model",
    "d_voice_state",
    "d_voice_state_explicit",
    "d_voice_state_ssl",
    "uclm_n_heads",
    "uclm_n_layers",
    "uclm_vq_bins",
    "uclm_dropout",
    "uclm_max_seq_len",
    "uclm_context_frames",
    "uclm_block_size",
    "codebook_dim",
    "latent_dim",
    "enc_state_dim",
    "enc_state_frames",
    "dec_state_dim",
    "dec_state_frames",
    "d_event_trace",
    "kv_cache_size",
    "d_f0",
    "f0_smoothing_frames",
    "d_suprasegmental",
    "cfg_scale_default",
    "cfg_scale_max",
    "max_prompt_seconds_active",
    "max_prompt_frames",
    "max_prompt_kv_tokens",
    "max_prompt_cache_bytes",
    "max_frames_per_unit",
    "skip_protection_threshold",
    "max_text_units_active",
    "max_dialogue_context_units",
    "max_acoustic_history_frames",
    "max_cross_attn_kv_bytes",
    "streaming_latency_budget_ms",
    "streaming_hardware_class_primary",
    "serve_port",
    "gui_port",
    "n_acting_tags",
    "extended_vocab_size",
    "d_acting_latent",
    "d_acting_macro",
    "bio_covariance_rank",
    "bio_transition_penalty_weight",
}

_RUST_NAME_MAP: dict[str, str] = {
    "content_encoder_state_frames": "CONTENT_ENC_STATE_FRAMES",
    "ir_estimator_state_frames": "IR_EST_STATE_FRAMES",
}


def _py_value(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        inner = ", ".join(_py_value(x) for x in v)
        return f"[{inner}]"
    return str(v)


def _generate_python(cfg: dict) -> str:
    lines = [
        '"""Auto-generated constants — DO NOT EDIT.',
        "",
        "Regenerated automatically on import when constants.yaml changes.",
        '"""',
        "",
    ]
    for key, val in cfg.items():
        name = key.upper()
        lines.append(f"{name} = {_py_value(val)}")
    return "\n".join(lines) + "\n"


def _rust_type(v: object) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, float):
        return "f32"
    if isinstance(v, int):
        return "usize"
    if isinstance(v, str):
        return "&str"
    return "usize"


def _rust_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        inner = ", ".join(_rust_value(x) for x in v)
        return f"[{inner}]"
    return str(v)


def _generate_rust(cfg: dict) -> str:
    sections = {
        "audio": [
            "sample_rate",
            "n_fft",
            "hop_length",
            "window_length",
            "n_mels",
            "mel_fmin",
            "mel_fmax",
            "n_freq_bins",
            "log_floor",
        ],
        "model": [
            "d_content",
            "d_speaker",
            "n_ir_params",
            "n_voice_source_params",
            "n_acoustic_params",
            "d_converter_hidden",
            "d_ir_estimator_hidden",
            "d_vocoder_hidden",
            "d_vocoder_features",
        ],
        "inference": ["ir_update_interval"],
        "lora": ["lora_rank", "lora_delta_size"],
        "state": [
            "content_encoder_state_frames",
            "ir_estimator_state_frames",
            "converter_state_frames",
            "vocoder_state_frames",
        ],
        "hq": [
            "max_lookahead_hops",
            "converter_hq_state_frames",
            "hq_threshold_q",
            "crossfade_frames",
        ],
        "tts": [
            "d_style",
            "n_style_params",
            "d_text_encoder",
            "n_text_encoder_layers",
            "n_text_encoder_heads",
            "text_encoder_ff_dim",
            "phoneme_vocab_size",
            "tokenizer_vocab_size",
            "n_emotion_categories",
            "d_f0_predictor",
            "d_content_synthesizer",
            "n_languages",
        ],
        "uclm": [
            "n_codebooks",
            "rvq_vocab_size",
            "control_vocab_size",
            "control_slots",
            "d_model",
            "d_voice_state",
            "d_voice_state_explicit",
            "d_voice_state_ssl",
            "uclm_n_heads",
            "uclm_n_layers",
            "uclm_vq_bins",
            "uclm_dropout",
            "uclm_max_seq_len",
            "uclm_context_frames",
            "uclm_block_size",
            "codebook_dim",
            "latent_dim",
            "enc_state_dim",
            "enc_state_frames",
            "dec_state_dim",
            "dec_state_frames",
            "d_event_trace",
        ],
        "f0": [
            "d_f0",
            "f0_smoothing_frames",
        ],
        "suprasegmental": [
            "d_suprasegmental",
        ],
        "cfg": [
            "cfg_scale_default",
            "cfg_scale_max",
        ],
        "prompt_budget": [
            "max_prompt_seconds_active",
            "max_prompt_frames",
            "max_prompt_kv_tokens",
            "max_prompt_cache_bytes",
        ],
        "runtime_budget": [
            "max_text_units_active",
            "max_dialogue_context_units",
            "max_acoustic_history_frames",
            "max_cross_attn_kv_bytes",
            "streaming_latency_budget_ms",
            "streaming_hardware_class_primary",
        ],
        "server": [
            "serve_port",
            "gui_port",
        ],
        "v4_acting": [
            "n_acting_tags",
            "extended_vocab_size",
            "d_acting_latent",
            "d_acting_macro",
            "bio_covariance_rank",
            "bio_transition_penalty_weight",
        ],
    }

    section_headers = {
        "audio": "// --- Audio parameters ---",
        "model": "\n// --- Model dimensions ---",
        "inference": "\n// --- Inference parameters ---",
        "lora": "\n// --- LoRA parameters ---",
        "state": "\n// --- State tensor context lengths ---",
        "hq": "\n// --- Lookahead / HQ mode ---",
        "tts": "\n// --- TTS extension ---",
        "uclm": "\n// --- UCLM (Unified Codec Language Model) ---",
        "f0": "\n// --- F0 Conditioning ---",
        "suprasegmental": "\n// --- Suprasegmental text features (v3) ---",
        "cfg": "\n// --- CFG (Classifier-Free Guidance) ---",
        "prompt_budget": "\n// --- Prompt budget limits (v3) ---",
        "runtime_budget": "\n// --- Runtime budget limits (v3) ---",
        "server": "\n// --- Server ports ---",
        "v4_acting": "\n// --- v4 Acting & Biological Constraints ---",
    }

    lines = [
        "// Auto-generated from configs/constants.yaml — DO NOT EDIT.",
        "",
        "#![allow(dead_code)]",
        "",
    ]

    for section, keys in sections.items():
        lines.append(section_headers[section])
        for key in keys:
            if key not in cfg:
                continue
            val = cfg[key]
            rust_name = _RUST_NAME_MAP.get(key, key.upper())
            rtype = _rust_type(val)
            rval = _rust_value(val)
            if isinstance(val, list):
                lines.append(
                    f"pub const {rust_name}: [{_rust_type(val[0])}; {len(val)}] = {rval};"
                )
            else:
                lines.append(f"pub const {rust_name}: {rtype} = {rval};")

    lines.append("")
    lines.append("// --- Derived constants ---")
    lines.append("pub const RING_BUFFER_CAPACITY: usize = 4096;")
    lines.append("pub const MAX_DAW_BLOCK_SIZE: usize = 4096;")
    lines.append("")
    lines.append("// Past context for causal windowing: WINDOW_LENGTH - HOP_LENGTH")
    lines.append("pub const PAST_CONTEXT: usize = WINDOW_LENGTH - HOP_LENGTH;")
    lines.append("")
    lines.append("// UCLM derived constants")
    lines.append("pub const FRAME_SIZE: usize = HOP_LENGTH;")
    lines.append("pub const FRAME_RATE: usize = 100;")
    lines.append("pub const CONTEXT_FRAMES: usize = UCLM_CONTEXT_FRAMES;")
    lines.append("pub const CONTEXT_LENGTH: usize = UCLM_CONTEXT_FRAMES;")
    lines.append("pub const CODEBOOK_SIZE: usize = RVQ_VOCAB_SIZE;")
    lines.append("pub const N_HEADS: usize = UCLM_N_HEADS;")
    lines.append("pub const N_LAYERS: usize = UCLM_N_LAYERS;")
    lines.append("pub const HEAD_DIM: usize = D_MODEL / N_HEADS;")
    lines.append(
        "pub const CODEC_ENCODER_STATE_SIZE: usize = ENC_STATE_DIM * ENC_STATE_FRAMES;"
    )
    lines.append(
        "pub const CODEC_DECODER_STATE_SIZE: usize = DEC_STATE_DIM * DEC_STATE_FRAMES;"
    )
    lines.append(
        "pub const KV_CACHE_SIZE: usize = UCLM_N_LAYERS * 2 * N_HEADS * UCLM_CONTEXT_FRAMES * HEAD_DIM;"
    )
    lines.append(
        "pub const CTX_A_BUFFER_SIZE: usize = UCLM_CONTEXT_FRAMES * N_CODEBOOKS;"
    )
    lines.append(
        "pub const CTX_B_BUFFER_SIZE: usize = UCLM_CONTEXT_FRAMES * CONTROL_SLOTS;"
    )
    lines.append("")

    return "\n".join(lines)


def _write_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def ensure_generated() -> None:
    """Regenerate constants if YAML is newer than generated files."""
    if not _YAML_PATH.exists():
        return

    yaml_mtime = _YAML_PATH.stat().st_mtime
    py_mtime = _PY_OUT.stat().st_mtime if _PY_OUT.exists() else 0
    rust_mtime = _RUST_OUT.stat().st_mtime if _RUST_OUT.exists() else 0

    if yaml_mtime <= py_mtime and yaml_mtime <= rust_mtime:
        return

    with open(_YAML_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    changed_py = _write_if_changed(_PY_OUT, _generate_python(cfg))
    changed_rust = _write_if_changed(_RUST_OUT, _generate_rust(cfg))

    if changed_py or changed_rust:
        print(f"Regenerated constants from {_YAML_PATH}")
