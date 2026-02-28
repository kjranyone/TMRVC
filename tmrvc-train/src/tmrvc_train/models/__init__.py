"""TMRVC model definitions (Codec-Latent pipeline)."""

from tmrvc_train.models.streaming_codec import (
    CodecConfig,
    StreamingCodec,
    StreamingCodecEncoder,
    StreamingCodecDecoder,
    ResidualVectorQuantizer,
    MultiScaleDiscriminator,
    MultiScaleSTFTLoss,
    create_streaming_codec,
)
from tmrvc_train.models.token_model import (
    TokenModelConfig,
    TokenModel,
    create_token_model,
)
from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA
from tmrvc_train.models.style_encoder import StyleEncoder
from tmrvc_train.models.ir_estimator import IREstimator

__all__ = [
    "CodecConfig",
    "StreamingCodec",
    "StreamingCodecEncoder",
    "StreamingCodecDecoder",
    "ResidualVectorQuantizer",
    "MultiScaleDiscriminator",
    "MultiScaleSTFTLoss",
    "create_streaming_codec",
    "TokenModelConfig",
    "TokenModel",
    "create_token_model",
    "SpeakerEncoderWithLoRA",
    "StyleEncoder",
    "IREstimator",
]
