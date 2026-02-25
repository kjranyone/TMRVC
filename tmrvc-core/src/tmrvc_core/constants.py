"""TMRVC shared constants loaded from configs/constants.yaml."""

from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).resolve().parents[3] / "configs" / "constants.yaml"

with open(_YAML_PATH, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# --- Audio Processing ---
SAMPLE_RATE: int = _cfg["sample_rate"]
N_FFT: int = _cfg["n_fft"]
HOP_LENGTH: int = _cfg["hop_length"]
WINDOW_LENGTH: int = _cfg["window_length"]
N_MELS: int = _cfg["n_mels"]
MEL_FMIN: float = _cfg["mel_fmin"]
MEL_FMAX: float = _cfg["mel_fmax"]
N_FREQ_BINS: int = _cfg["n_freq_bins"]
LOG_FLOOR: float = _cfg["log_floor"]

# --- Model Dimensions ---
D_CONTENT: int = _cfg["d_content"]
D_SPEAKER: int = _cfg["d_speaker"]
N_IR_PARAMS: int = _cfg["n_ir_params"]
N_VOICE_SOURCE_PARAMS: int = _cfg["n_voice_source_params"]
N_ACOUSTIC_PARAMS: int = _cfg["n_acoustic_params"]
D_CONVERTER_HIDDEN: int = _cfg["d_converter_hidden"]
D_IR_ESTIMATOR_HIDDEN: int = _cfg["d_ir_estimator_hidden"]
D_VOCODER_HIDDEN: int = _cfg["d_vocoder_hidden"]
D_VOCODER_FEATURES: int = _cfg["d_vocoder_features"]

# --- Feature Extraction ---
D_CONTENT_VEC: int = _cfg["d_content_vec"]
CONTENT_VEC_HOP_MS: float = _cfg["content_vec_hop_ms"]

# --- WavLM (Phase 1+) ---
D_WAVLM_LARGE: int = _cfg["d_wavlm_large"]
WAVLM_LAYER: int = _cfg["wavlm_layer"]

# --- VQ Bottleneck ---
VQ_N_CODEBOOKS: int = _cfg["vq_n_codebooks"]
VQ_CODEBOOK_SIZE: int = _cfg["vq_codebook_size"]
VQ_CODEBOOK_DIM: int = _cfg["vq_codebook_dim"]
VQ_COMMITMENT_LAMBDA: float = _cfg["vq_commitment_lambda"]

# --- OT-CFM ---
USE_OT_CFM: bool = _cfg["use_ot_cfm"]
OT_CFM_BATCH_OT: bool = _cfg["ot_cfm_batch_ot"]

# --- Quality Targets ---
TARGET_SECS_PHASE0: float = _cfg["target_secs_phase0"]
TARGET_SECS_PHASE1: float = _cfg["target_secs_phase1"]
TARGET_UTMOS_PHASE1: float = _cfg["target_utmos_phase1"]
TARGET_SECS_REVERB: float = _cfg["target_secs_reverb"]

# --- Inference Parameters ---
STUDENT_STEPS: int = _cfg["student_steps"]
IR_UPDATE_INTERVAL: int = _cfg["ir_update_interval"]

# --- LoRA Parameters ---
LORA_RANK: int = _cfg["lora_rank"]
LORA_ALPHA: int = _cfg["lora_alpha"]
N_LORA_LAYERS: int = _cfg["n_lora_layers"]
LORA_DELTA_SIZE: int = _cfg["lora_delta_size"]

# --- State Tensor Dimensions ---
CONTENT_ENCODER_STATE_FRAMES: int = _cfg["content_encoder_state_frames"]
IR_ESTIMATOR_STATE_FRAMES: int = _cfg["ir_estimator_state_frames"]
CONVERTER_STATE_FRAMES: int = _cfg["converter_state_frames"]
VOCODER_STATE_FRAMES: int = _cfg["vocoder_state_frames"]

# --- IR Subbands ---
N_IR_SUBBANDS: int = _cfg["n_ir_subbands"]
IR_SUBBAND_EDGES_HZ: list[int] = _cfg["ir_subband_edges_hz"]

# --- Lookahead / HQ Mode ---
MAX_LOOKAHEAD_HOPS: int = _cfg["max_lookahead_hops"]
CONVERTER_HQ_STATE_FRAMES: int = _cfg["converter_hq_state_frames"]
HQ_THRESHOLD_Q: float = _cfg["hq_threshold_q"]
CROSSFADE_FRAMES: int = _cfg["crossfade_frames"]

# --- Global Timbre Memory ---
GTM_N_ENTRIES: int = _cfg["gtm_n_entries"]
GTM_D_ENTRY: int = _cfg["gtm_d_entry"]
GTM_N_HEADS: int = _cfg["gtm_n_heads"]

# --- Teacher Architecture ---
D_TEACHER_HIDDEN: int = _cfg["d_teacher_hidden"]
TEACHER_DOWN_CHANNELS: list[int] = _cfg["teacher_down_channels"]
TEACHER_N_HEADS: int = _cfg["teacher_n_heads"]

# --- Training ---
FLOW_MATCHING_STEPS: int = _cfg["flow_matching_steps"]

# --- Training Defaults ---
DEFAULT_BATCH_SIZE: int = _cfg["default_batch_size"]
SEGMENT_MIN_SEC: float = _cfg["segment_min_sec"]
SEGMENT_MAX_SEC: float = _cfg["segment_max_sec"]
LOUDNESS_TARGET_LUFS: float = _cfg["loudness_target_lufs"]
CROSS_SPEAKER_PROB: float = _cfg["cross_speaker_prob"]

# --- Voice Source Param Names ---
VOICE_SOURCE_PARAM_NAMES: list[str] = _cfg["voice_source_param_names"]

# --- TTS Extension ---
D_STYLE: int = _cfg["d_style"]
N_STYLE_PARAMS: int = _cfg["n_style_params"]
D_TEXT_ENCODER: int = _cfg["d_text_encoder"]
N_TEXT_ENCODER_LAYERS: int = _cfg["n_text_encoder_layers"]
N_TEXT_ENCODER_HEADS: int = _cfg["n_text_encoder_heads"]
TEXT_ENCODER_FF_DIM: int = _cfg["text_encoder_ff_dim"]
PHONEME_VOCAB_SIZE: int = _cfg["phoneme_vocab_size"]
TOKENIZER_VOCAB_SIZE: int = _cfg["tokenizer_vocab_size"]
N_EMOTION_CATEGORIES: int = _cfg["n_emotion_categories"]
D_F0_PREDICTOR: int = _cfg["d_f0_predictor"]
D_CONTENT_SYNTHESIZER: int = _cfg["d_content_synthesizer"]
N_LANGUAGES: int = _cfg["n_languages"]

# --- Scene State Latent (SSL) ---
D_SCENE_STATE: int = _cfg["d_scene_state"]
D_HISTORY: int = _cfg["d_history"]
SSL_N_GRU_LAYERS: int = _cfg["ssl_n_gru_layers"]
SSL_PROSODY_STATS_DIM: int = _cfg["ssl_prosody_stats_dim"]

# --- Breath-Pause Event Head (BPEH) ---
BPEH_D_HIDDEN: int = _cfg["bpeh_d_hidden"]
BPEH_N_BLOCKS: int = _cfg["bpeh_n_blocks"]
BPEH_BREATH_THRESHOLD_DB: float = _cfg["bpeh_breath_threshold_db"]
BPEH_MIN_PAUSE_MS: float = _cfg["bpeh_min_pause_ms"]
BPEH_MIN_BREATH_MS: float = _cfg["bpeh_min_breath_ms"]
