"""Superintelligence research toolkit.

This module exposes the main components required to configure and train the
reference model defined in :mod:`superintelligence.model`.
"""

from .config import ModelConfig, OptimizerConfig, TrainingConfig
from .model import SuperIntelligenceModel
from .trainer import Trainer

__all__ = [
    "ModelConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "SuperIntelligenceModel",
    "Trainer",
]
