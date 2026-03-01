"""UCLM model definitions (Unified Codec Language Model)."""

from tmrvc_train.models.uclm import VectorQuantizer, VCEncoder
from tmrvc_train.models.uclm_transformer import CodecTransformer
from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_train.models.uclm_loss import uclm_loss
from tmrvc_train.models.duration_predictor import DurationPredictor, duration_loss
from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA
from tmrvc_train.models.text_encoder import TextEncoder
from tmrvc_train.models.text_features import (
    TextFeatureExpander,
    expand_phonemes_to_frames,
)
from tmrvc_train.models.voice_state_film import VoiceStateFiLM, MultiVoiceStateFiLM
from tmrvc_train.models.voice_state_encoder import (
    VoiceStateEncoder,
    VoiceStateEncoderForStreaming,
    create_voice_state_encoder,
)
from tmrvc_train.models.ssl_extractor import (
    SSLProjection,
    WavLMSSLExtractor,
    StreamingSSLExtractor,
    MockSSLExtractor,
    create_ssl_extractor,
)
from tmrvc_train.models.control_encoder import (
    ControlEncoder,
    ControlEncoderTemporal,
    CONTROL_VOCAB,
    encode_duration_bin,
    encode_intensity_bin,
    decode_duration_bin,
    decode_intensity_bin,
)
from tmrvc_train.models.control_tokenizer import (
    ControlTokenizer,
    ControlEvent,
    EventTrace,
    event_to_tuple,
    tuple_to_event,
)
from tmrvc_train.models.emotion_codec import (
    EmotionAwareCodec,
    EmotionAwareEncoder,
    EmotionAwareDecoder,
    CodecLoss,
    multiscale_stft_loss,
)
from tmrvc_train.models.disentangle_losses import (
    DisentanglementLoss,
    GradientReversalLayer,
    orthogonality_loss,
    transition_smoothness_loss,
    breath_energy_coupling_loss,
    delta_state_consistency_loss,
    long_event_consistency_loss,
    duration_calibration_loss,
)

__all__ = [
    # UCLM Components
    "VoiceStateEncoder",
    "VoiceStateEncoderForStreaming",
    "create_voice_state_encoder",
    "VectorQuantizer",
    "VCEncoder",
    "CodecTransformer",
    "DisentangledUCLM",
    "uclm_loss",
    "DurationPredictor",
    "duration_loss",
    # SSL Extractor
    "SSLProjection",
    "WavLMSSLExtractor",
    "StreamingSSLExtractor",
    "MockSSLExtractor",
    "create_ssl_extractor",
    # Encoders
    "SpeakerEncoderWithLoRA",
    "TextEncoder",
    "TextFeatureExpander",
    "expand_phonemes_to_frames",
    # Emotion-Aware Codec (Token Spec v2)
    "VoiceStateFiLM",
    "MultiVoiceStateFiLM",
    "ControlEncoder",
    "ControlEncoderTemporal",
    "CONTROL_VOCAB",
    "encode_duration_bin",
    "encode_intensity_bin",
    "decode_duration_bin",
    "decode_intensity_bin",
    "ControlTokenizer",
    "ControlEvent",
    "EventTrace",
    "event_to_tuple",
    "tuple_to_event",
    "EmotionAwareCodec",
    "EmotionAwareEncoder",
    "EmotionAwareDecoder",
    "CodecLoss",
    "multiscale_stft_loss",
    # Disentanglement Losses
    "DisentanglementLoss",
    "GradientReversalLayer",
    "orthogonality_loss",
    "transition_smoothness_loss",
    "breath_energy_coupling_loss",
    "delta_state_consistency_loss",
    "long_event_consistency_loss",
    "duration_calibration_loss",
]
