"""Configuration dataclasses used across the SuperIntelligence stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Hyperparameters that define the model architecture."""

    vocab_size: int = 64_000
    max_sequence_length: int = 4_096
    embedding_dim: int = 2_048
    num_layers: int = 24
    num_heads: int = 16
    feedforward_dim: int = 8_192
    rotary_dim: int = 32
    dropout: float = 0.1
    expert_capacity: int = 4
    num_experts: int = 8


@dataclass
class OptimizerConfig:
    """Hyperparameters for the optimizer."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 2_000
    max_gradient_norm: float = 1.0


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    total_steps: int = 100_000
    micro_batch_size: int = 4
    global_batch_size: int = 256
    eval_interval: int = 1_000
    checkpoint_interval: int = 5_000
    logging_interval: int = 100
    devices: List[str] = field(default_factory=lambda: ["cuda:0"])
    precision: str = "bf16"
