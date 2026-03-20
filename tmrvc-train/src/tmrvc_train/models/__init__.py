"""UCLM model definitions (Unified Codec Language Model)."""

from tmrvc_train.models.uclm_model import (
    VectorQuantizer,
    VCEncoder,
    DisentangledUCLM,
    DialogueContextProjector,
    PointerHead,
    SpeakerPromptEncoder,
    ProsodyPredictor,
)
from tmrvc_train.models.uclm_transformer import CodecTransformer
from tmrvc_train.models.uclm_loss import uclm_loss
from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA
from tmrvc_train.models.text_encoder import TextEncoder
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
)
from tmrvc_train.models.reference_encoder import (
    ReferenceEncoder,
    ReferenceEncoderFromWaveform,
)
from tmrvc_train.models.vocoder import (
    VocoderBase,
    CodecNativeDecoder,
    VocosDecoder,
    HiFiGANDecoder,
    create_vocoder,
)
from tmrvc_train.models.disentangle_losses import (
    DisentanglementLoss,
    GradientReversalLayer,
    orthogonality_loss,
    transition_smoothness_loss,
    breath_energy_coupling_loss,
    delta_state_consistency_loss,
    long_event_consistency_loss,
)
from tmrvc_train.models.acting_latent import (
    ActingLatentEncoder,
    ActingLatentPredictor,
    ActingMacroProjector,
    ActingLatentConditioner,
)
from tmrvc_train.models.biological_constraints import (
    BiologicalConstraintRegularizer,
)
from tmrvc_train.models.acting_losses import (
    acting_latent_kl_loss,
    acting_latent_usage_loss,
    disentanglement_loss,
    semantic_alignment_loss,
)

__all__ = [
    # UCLM Components
    "VoiceStateEncoder",
    "VoiceStateEncoderForStreaming",
    "create_voice_state_encoder",
    "VectorQuantizer",
    "VCEncoder",
    "CodecTransformer",
    "DialogueContextProjector",
    "DisentangledUCLM",
    "PointerHead",
    "SpeakerPromptEncoder",
    "ProsodyPredictor",
    "uclm_loss",
    # SSL Extractor
    "SSLProjection",
    "WavLMSSLExtractor",
    "StreamingSSLExtractor",
    "MockSSLExtractor",
    "create_ssl_extractor",
    # Encoders
    "SpeakerEncoderWithLoRA",
    "TextEncoder",
    # Emotion-Aware Codec (Token Spec)
    "VoiceStateFiLM",
    "MultiVoiceStateFiLM",
    "ControlEncoder",
    "ControlEncoderTemporal",
    "CONTROL_VOCAB",
    "ControlTokenizer",
    "ControlEvent",
    "EventTrace",
    "event_to_tuple",
    "tuple_to_event",
    "EmotionAwareCodec",
    "EmotionAwareEncoder",
    "EmotionAwareDecoder",
    # Disentanglement Losses
    "DisentanglementLoss",
    "GradientReversalLayer",
    "orthogonality_loss",
    "transition_smoothness_loss",
    "breath_energy_coupling_loss",
    "delta_state_consistency_loss",
    "long_event_consistency_loss",
    # Reference Encoder (GST-style prosody extraction)
    "ReferenceEncoder",
    "ReferenceEncoderFromWaveform",
    # Vocoder Interface
    "VocoderBase",
    "CodecNativeDecoder",
    "VocosDecoder",
    "HiFiGANDecoder",
    "create_vocoder",
    # Acting Latent (v4)
    "ActingLatentEncoder",
    "ActingLatentPredictor",
    "ActingMacroProjector",
    "ActingLatentConditioner",
    # Biological Constraints (v4)
    "BiologicalConstraintRegularizer",
    # Acting Losses (v4)
    "acting_latent_kl_loss",
    "acting_latent_usage_loss",
    "disentanglement_loss",
    "semantic_alignment_loss",
]
