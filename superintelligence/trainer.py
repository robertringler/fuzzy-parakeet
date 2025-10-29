"""Training loop utilities."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn, optim

from .config import ModelConfig, OptimizerConfig, TrainingConfig
from .dataset import chunk_iterator
from .model import SuperIntelligenceModel


@dataclass
class TrainerState:
    global_step: int = 0
    best_loss: float = float("inf")


class Trainer:
    """Minimal training harness supporting automatic mixed precision."""

    def __init__(
        self,
        model: Optional[SuperIntelligenceModel] = None,
        model_config: Optional[ModelConfig] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model or SuperIntelligenceModel(model_config)
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.training_config = training_config or TrainingConfig()
        self.state = TrainerState()

        self.device = device or torch.device(self.training_config.devices[0])
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.betas,
            eps=self.optimizer_config.eps,
            weight_decay=self.optimizer_config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=max(self.optimizer_config.warmup_steps, 1),
        )

    def _autocast(self):
        if self.training_config.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def train(
        self,
        data_iterator: Iterable[torch.Tensor],
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        loss_fn = loss_fn or nn.CrossEntropyLoss()
        cfg = self.training_config
        micro_batch_size = cfg.micro_batch_size
        accumulation_steps = max(1, cfg.global_batch_size // micro_batch_size)

        for batch in chunk_iterator(data_iterator, micro_batch_size):
            self.model.train()
            batch = batch.to(self.device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            with self._autocast():
                logits = self.model(inputs)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            (loss / accumulation_steps).backward()

            if (self.state.global_step + 1) % accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.optimizer_config.max_gradient_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            if self.state.global_step % cfg.logging_interval == 0:
                print(f"step={self.state.global_step} loss={loss.item():.4f}")

            self.state.global_step += 1
            if self.state.global_step >= cfg.total_steps:
                break
