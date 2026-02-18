"""TMRVC model definitions."""

from tmrvc_train.models.content_encoder import ContentEncoderStudent
from tmrvc_train.models.converter import (
    ConverterStudent,
    ConverterStudentGTM,
    ConverterStudentHQ,
)
from tmrvc_train.models.discriminator import MelDiscriminator
from tmrvc_train.models.ir_estimator import IREstimator
from tmrvc_train.models.vocoder import VocoderStudent

__all__ = [
    "ContentEncoderStudent",
    "ConverterStudent",
    "ConverterStudentGTM",
    "ConverterStudentHQ",
    "IREstimator",
    "MelDiscriminator",
    "VocoderStudent",
]
