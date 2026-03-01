"""TMRVC shared constants loaded from configs/constants.yaml (UCLM v2)."""

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
D_MODEL: int = _cfg["d_model"]
D_CONTENT: int = _cfg["d_content"]
D_CONTENT_VEC: int = _cfg["d_content_vec"]
D_SPEAKER: int = _cfg["d_speaker"]
D_F0: int = _cfg["d_f0"]

# --- UCLM v2 Token Spec ---
N_CODEBOOKS: int = _cfg["n_codebooks"]
RVQ_VOCAB_SIZE: int = _cfg["rvq_vocab_size"]
CONTROL_SLOTS: int = _cfg["control_slots"]
CONTROL_VOCAB_SIZE: int = _cfg["control_vocab_size"]

# --- Voice State ---
D_VOICE_STATE: int = _cfg["d_voice_state"]
D_VOICE_STATE_SSL: int = _cfg["d_voice_state_ssl"]

# --- VQ Bottleneck ---
UCLM_VQ_BINS: int = _cfg["uclm_vq_bins"]

# --- Text / TTS ---
D_TEXT_ENCODER: int = _cfg["d_text_encoder"]
N_TEXT_ENCODER_LAYERS: int = _cfg["n_text_encoder_layers"]
N_TEXT_ENCODER_HEADS: int = _cfg["n_text_encoder_heads"]
TEXT_ENCODER_FF_DIM: int = _cfg["text_encoder_ff_dim"]
PHONEME_VOCAB_SIZE: int = _cfg["phoneme_vocab_size"]
N_LANGUAGES: int = _cfg["n_languages"]
N_VOICE_SOURCE_PARAMS: int = _cfg["n_voice_source_params"]
N_ACOUSTIC_PARAMS: int = _cfg["n_acoustic_params"]
D_STYLE: int = _cfg["d_style"]
N_STYLE_PARAMS: int = _cfg["n_style_params"]

# --- LoRA ---
LORA_RANK: int = _cfg["lora_rank"]
LORA_ALPHA: int = _cfg["lora_alpha"]
LORA_DELTA_SIZE: int = _cfg["lora_delta_size"]

# --- Scene State ---
D_SCENE_STATE: int = _cfg["d_scene_state"]

# --- Training / Inference ---
UCLM_N_HEADS: int = _cfg["uclm_n_heads"]
UCLM_N_LAYERS: int = _cfg["uclm_n_layers"]
UCLM_MAX_SEQ_LEN: int = _cfg["uclm_max_seq_len"]
UCLM_CONTEXT_FRAMES: int = _cfg["uclm_context_frames"]
UCLM_BLOCK_SIZE: int = _cfg["uclm_block_size"]

# --- Legacy Aliases (kept for minor compatibility, but should be avoided) ---
VOCAB_SIZE: int = RVQ_VOCAB_SIZE
