"""A small Transformer encoder implemented from scratch in PyTorch.

Nothing here is imported from transformers / sentence-transformers. We build:

* :class:`MultiHeadSelfAttention` — scaled dot-product attention with an
  additive padding mask, implemented with explicit Q/K/V projections.
* :class:`FeedForward` — the position-wise MLP (GELU).
* :class:`TransformerBlock` — pre-LayerNorm residual block (attn + FFN).
* :class:`TransformerEncoder` — token + learned-positional + optional segment
  embeddings, a stack of blocks, and a final LayerNorm.

The module is import-guarded by its callers: this file imports torch at module
load, so it is only imported on the training/ML path (never by the min CI path).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EncoderConfig", "TransformerEncoder", "mean_pool", "MultiHeadSelfAttention"]


@dataclass
class EncoderConfig:
    vocab_size: int = 8000
    max_len: int = 192
    dim: int = 256
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    n_segments: int = 2  # token-type embeddings (used by the cross-encoder)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention with a padding mask."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D); attn_mask: (B, T) with 1 for real tokens, 0 for pad.
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        # Additive mask: -inf on padded *key* positions.
        mask = (attn_mask == 0).view(b, 1, 1, t)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        # Rows that are fully masked (padded queries) become NaN-free zeros.
        attn = torch.nan_to_num(attn)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm residual block."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = MultiHeadSelfAttention(cfg.dim, cfg.heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.ffn = FeedForward(cfg.dim, cfg.mlp_ratio, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Token + positional (+ segment) embeddings -> N blocks -> final LayerNorm."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.dim)
        self.seg_emb = nn.Embedding(cfg.n_segments, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        if segment_ids is not None:
            x = x + self.seg_emb(segment_ids)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.norm(x)  # (B, T, D)


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token states over real (non-pad) positions."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts
