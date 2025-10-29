"""High-level model definition."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .config import ModelConfig
from .modules import RMSNorm, TransformerBlock


class SuperIntelligenceModel(nn.Module):
    """Transformer-like language model with sparse MoE experts."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, cfg.max_sequence_length, cfg.embedding_dim)
        )
        self.dropout = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.embedding_dim,
                    num_heads=cfg.num_heads,
                    rotary_dim=cfg.rotary_dim,
                    max_sequence_length=cfg.max_sequence_length,
                    feedforward_dim=cfg.feedforward_dim,
                    dropout=cfg.dropout,
                    num_experts=cfg.num_experts,
                    capacity_factor=cfg.expert_capacity,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = RMSNorm(cfg.embedding_dim)
        self.lm_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del attention_mask  # Masking policy is handled implicitly via caching.

        seq_len = input_ids.size(1)
        position_embeddings = self.position_embedding[:, :seq_len, :]
        hidden_states = self.token_embedding(input_ids) + position_embeddings
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Simple autoregressive sampling loop."""

        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
