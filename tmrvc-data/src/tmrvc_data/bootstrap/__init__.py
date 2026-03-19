"""v4 raw-audio bootstrap pipeline.

Converts raw unlabeled audio corpora into train-ready cache
following the v4 dataset contract.
"""

from tmrvc_data.bootstrap.pipeline import BootstrapPipeline
from tmrvc_data.bootstrap.contracts import (
    BootstrapStage,
    BootstrapConfig,
    BootstrapResult,
    TrainReadyCacheContract,
)
from tmrvc_data.bootstrap.supervision import SupervisionTierClassifier

__all__ = [
    "BootstrapPipeline",
    "BootstrapStage",
    "BootstrapConfig",
    "BootstrapResult",
    "TrainReadyCacheContract",
    "SupervisionTierClassifier",
]
