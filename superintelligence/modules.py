"""Model building blocks for the SuperIntelligence architecture."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding implementation."""

    def __init__(self, dim: int, max_position: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(-2)
        cos = self.cos_cached[:seq_len, :].to(x.device)
        sin = self.sin_cached[:seq_len, :].to(x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Applies rotary positional embeddings to the query/key tensors."""

    return (x * cos) + (_rotate_half(x) * sin)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with rotary embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        rotary_dim: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rotary_dim = rotary_dim

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(rotary_dim, max_sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rotary(x)
        q_rot = torch.cat((
            apply_rotary_pos_emb(q[..., : self.rotary_dim], cos, sin),
            q[..., self.rotary_dim :],
        ), dim=-1)
        k_rot = torch.cat((
            apply_rotary_pos_emb(k[..., : self.rotary_dim], cos, sin),
            k[..., self.rotary_dim :],
        ), dim=-1)

        attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, dim)
        attn_output = self.out_proj(attn_output)
        return self.resid_dropout(attn_output)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.w1(x)) * self.w2(x)
        hidden = self.dropout(hidden)
        return self.w3(hidden)


class Expert(nn.Module):
    """Single expert used in the mixture-of-experts layer."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.ffn = SwiGLU(dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class SparseMoE(nn.Module):
    """Top-``k`` gating sparse mixture-of-experts layer."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        capacity_factor: int,
        dropout: float,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.k = k

        self.experts = nn.ModuleList(
            [Expert(dim, hidden_dim, dropout) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.k, dim=-1)
        gate_scores = torch.softmax(weights, dim=-1)

        output = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(x)
            weight = torch.zeros_like(gate_scores[..., :1])
            for top_idx in range(self.k):
                mask = (indices[..., top_idx : top_idx + 1] == expert_idx).float()
                weight = weight + gate_scores[..., top_idx : top_idx + 1] * mask
            output = output + expert_output * weight
        return output


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MoE feed-forward."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rotary_dim: int,
        max_sequence_length: int,
        feedforward_dim: int,
        dropout: float,
        num_experts: int,
        capacity_factor: int,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            rotary_dim=rotary_dim,
            max_sequence_length=max_sequence_length,
        )
        self.norm2 = RMSNorm(dim)
        self.moe = SparseMoE(
            dim=dim,
            hidden_dim=feedforward_dim,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x
