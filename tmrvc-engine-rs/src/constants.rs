// Auto-generated from configs/constants.yaml — DO NOT EDIT.

#![allow(dead_code)]

// --- Audio parameters ---
pub const SAMPLE_RATE: usize = 24000;
pub const N_FFT: usize = 1024;
pub const HOP_LENGTH: usize = 240;
pub const WINDOW_LENGTH: usize = 960;
pub const N_MELS: usize = 80;
pub const MEL_FMIN: f32 = 0.0;
pub const MEL_FMAX: f32 = 12000.0;
pub const N_FREQ_BINS: usize = 513;
pub const LOG_FLOOR: f32 = 1e-10;

// --- Model dimensions ---
pub const D_SPEAKER: usize = 192;
pub const N_VOICE_SOURCE_PARAMS: usize = 12;
pub const N_ACOUSTIC_PARAMS: usize = 32;

// --- Inference parameters ---

// --- LoRA parameters ---
pub const LORA_RANK: usize = 4;
pub const LORA_DELTA_SIZE: usize = 32768;

// --- State tensor context lengths ---

// --- Lookahead / HQ mode ---
pub const MAX_LOOKAHEAD_HOPS: usize = 6;
pub const HQ_THRESHOLD_Q: f32 = 0.3;

// --- TTS extension ---
pub const D_STYLE: usize = 128;
pub const D_TEXT_ENCODER: usize = 256;
pub const N_TEXT_ENCODER_LAYERS: usize = 6;
pub const N_TEXT_ENCODER_HEADS: usize = 4;
pub const TEXT_ENCODER_FF_DIM: usize = 1024;
pub const PHONEME_VOCAB_SIZE: usize = 200;
pub const N_LANGUAGES: usize = 4;

// --- UCLM (Unified Codec Language Model) ---
pub const N_CODEBOOKS: usize = 8;
pub const RVQ_VOCAB_SIZE: usize = 1024;
pub const CONTROL_VOCAB_SIZE: usize = 64;
pub const CONTROL_SLOTS: usize = 4;
pub const D_MODEL: usize = 768;
pub const D_VOICE_STATE: usize = 12;
pub const D_VOICE_STATE_EXPLICIT: usize = 12;
pub const D_VOICE_STATE_SSL: usize = 128;
pub const UCLM_N_HEADS: usize = 12;
pub const UCLM_N_LAYERS: usize = 16;
pub const UCLM_VQ_BINS: usize = 128;
pub const UCLM_DROPOUT: f32 = 0.1;
pub const UCLM_MAX_SEQ_LEN: usize = 2048;
pub const UCLM_CONTEXT_FRAMES: usize = 200;
pub const UCLM_BLOCK_SIZE: usize = 4;
pub const CODEBOOK_DIM: usize = 128;
pub const LATENT_DIM: usize = 768;
pub const ENC_STATE_DIM: usize = 768;
pub const ENC_STATE_FRAMES: usize = 32;
pub const DEC_STATE_DIM: usize = 256;
pub const DEC_STATE_FRAMES: usize = 32;
pub const D_EVENT_TRACE: usize = 64;

// --- F0 Conditioning ---
pub const D_F0: usize = 2;
pub const F0_SMOOTHING_FRAMES: usize = 5;

// --- Suprasegmental text features (v3) ---
pub const D_SUPRASEGMENTAL: usize = 4;

// --- CFG (Classifier-Free Guidance) ---
pub const CFG_SCALE_DEFAULT: f32 = 1.5;
pub const CFG_SCALE_MAX: f32 = 3.0;

// --- Prompt budget limits (v3) ---
pub const MAX_PROMPT_SECONDS_ACTIVE: f32 = 10.0;
pub const MAX_PROMPT_FRAMES: usize = 1000;
pub const MAX_PROMPT_KV_TOKENS: usize = 512;
pub const MAX_PROMPT_CACHE_BYTES: usize = 52428800;

// --- Runtime budget limits (v3) ---
pub const MAX_TEXT_UNITS_ACTIVE: usize = 512;
pub const MAX_DIALOGUE_CONTEXT_UNITS: usize = 1024;
pub const MAX_ACOUSTIC_HISTORY_FRAMES: usize = 2000;
pub const MAX_CROSS_ATTN_KV_BYTES: usize = 104857600;
pub const STREAMING_LATENCY_BUDGET_MS: f32 = 10.0;
pub const STREAMING_HARDWARE_CLASS_PRIMARY: &str = "single_nvidia_rtx_2080ti_22gb_cuda12_sdpa";

// --- Server ports ---
pub const SERVE_PORT: usize = 8100;
pub const GUI_PORT: usize = 7960;

// --- v4 Acting & Biological Constraints ---
pub const N_ACTING_TAGS: usize = 35;
pub const EXTENDED_VOCAB_SIZE: usize = 235;
pub const D_ACTING_LATENT: usize = 24;
pub const D_ACTING_MACRO: usize = 6;
pub const BIO_COVARIANCE_RANK: usize = 8;
pub const BIO_TRANSITION_PENALTY_WEIGHT: f32 = 0.1;

// --- Derived constants ---
pub const RING_BUFFER_CAPACITY: usize = 4096;
pub const MAX_DAW_BLOCK_SIZE: usize = 4096;

// Past context for causal windowing: WINDOW_LENGTH - HOP_LENGTH
pub const PAST_CONTEXT: usize = WINDOW_LENGTH - HOP_LENGTH;

// UCLM derived constants
pub const FRAME_SIZE: usize = HOP_LENGTH;
pub const FRAME_RATE: usize = 100;
pub const CONTEXT_FRAMES: usize = UCLM_CONTEXT_FRAMES;
pub const CONTEXT_LENGTH: usize = UCLM_CONTEXT_FRAMES;
pub const CODEBOOK_SIZE: usize = RVQ_VOCAB_SIZE;
pub const N_HEADS: usize = UCLM_N_HEADS;
pub const N_LAYERS: usize = UCLM_N_LAYERS;
pub const HEAD_DIM: usize = D_MODEL / N_HEADS;
pub const CODEC_ENCODER_STATE_SIZE: usize = ENC_STATE_DIM * ENC_STATE_FRAMES;
pub const CODEC_DECODER_STATE_SIZE: usize = DEC_STATE_DIM * DEC_STATE_FRAMES;
pub const KV_CACHE_SIZE: usize = UCLM_N_LAYERS * 2 * N_HEADS * UCLM_CONTEXT_FRAMES * HEAD_DIM;
pub const CTX_A_BUFFER_SIZE: usize = UCLM_CONTEXT_FRAMES * N_CODEBOOKS;
pub const CTX_B_BUFFER_SIZE: usize = UCLM_CONTEXT_FRAMES * CONTROL_SLOTS;
