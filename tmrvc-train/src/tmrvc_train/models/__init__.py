"""TMRVC model definitions."""

from tmrvc_train.models.content_encoder import ContentEncoderStudent
from tmrvc_train.models.content_synthesizer import ContentSynthesizer
from tmrvc_train.models.converter import (
    ConverterStudent,
    ConverterStudentGTM,
    ConverterStudentHQ,
    converter_from_vc_checkpoint,
)
from tmrvc_train.models.discriminator import MelDiscriminator
from tmrvc_train.models.duration_predictor import DurationPredictor
from tmrvc_train.models.f0_predictor import F0Predictor
from tmrvc_train.models.ir_estimator import IREstimator
from tmrvc_train.models.style_encoder import StyleEncoder
from tmrvc_train.models.text_encoder import TextEncoder
from tmrvc_train.models.vocoder import VocoderStudent

__all__ = [
    "ContentEncoderStudent",
    "ContentSynthesizer",
    "ConverterStudent",
    "ConverterStudentGTM",
    "ConverterStudentHQ",
    "DurationPredictor",
    "F0Predictor",
    "IREstimator",
    "MelDiscriminator",
    "StyleEncoder",
    "TextEncoder",
    "VocoderStudent",
    "converter_from_vc_checkpoint",
]
