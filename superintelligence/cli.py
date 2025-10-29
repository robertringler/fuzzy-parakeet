"""Command line interface for quick experiments."""

from __future__ import annotations

import argparse

import torch

from .config import ModelConfig, TrainingConfig
from .dataset import SyntheticTextDataset
from .trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SuperIntelligence model")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_argument(
        "--sequence-length", type=int, default=128, help="Synthetic sequence length"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = ModelConfig(max_sequence_length=args.sequence_length)
    training_config = TrainingConfig(total_steps=args.steps, devices=[args.device])

    dataset = SyntheticTextDataset(
        vocab_size=model_config.vocab_size,
        sequence_length=args.sequence_length,
        device=torch.device(args.device),
    )

    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        device=torch.device(args.device),
    )

    trainer.train(iter(dataset))


if __name__ == "__main__":
    main()
