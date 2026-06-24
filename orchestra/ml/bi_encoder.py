"""A bi-encoder: a from-scratch Transformer encoder + mean-pool -> L2-normalized
embedding, trained with InfoNCE (in-batch negatives).

Query and passage are encoded by the *same* tower (a shared encoder), which is a
standard and parameter-efficient choice for a small model. The InfoNCE loss with
in-batch negatives treats every other passage in the batch as a negative for a
given query.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from orchestra.ml.tokenizer import BPETokenizer
from orchestra.ml.transformer import EncoderConfig, TransformerEncoder, mean_pool

__all__ = ["BiEncoder", "info_nce_loss"]


class BiEncoder(nn.Module):
    """Shared-tower bi-encoder producing unit-norm embeddings."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerEncoder(cfg)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)
        pooled = mean_pool(hidden, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    # -- inference helpers -------------------------------------------------
    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        tokenizer: BPETokenizer,
        *,
        device: "torch.device | str" = "cpu",
        max_len: int = 128,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Encode raw texts into a ``(n, dim)`` tensor of unit-norm embeddings."""
        self.eval()
        out: List[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            ids, mask = tokenizer.encode_batch(batch, max_len=max_len)
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            mask_t = torch.tensor(mask, dtype=torch.long, device=device)
            emb = self.forward(ids_t, mask_t)
            out.append(emb.detach().to("cpu"))
        if not out:
            return torch.zeros((0, self.cfg.dim))
        return torch.cat(out, dim=0)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict()}, str(path))

    @classmethod
    def load(cls, path: str | Path, *, map_location: "str | torch.device" = "cpu") -> "BiEncoder":
        # weights_only=False is required to restore the EncoderConfig dataclass
        # stored alongside the weights. These are first-party checkpoints written
        # by this package (see save()), never untrusted input. nosec B614.
        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)  # nosec B614
        cfg = EncoderConfig(**ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


def info_nce_loss(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE with in-batch negatives.

    ``q_emb`` and ``p_emb`` are ``(B, D)`` unit-norm. The positive for query ``i``
    is passage ``i``; every other passage in the batch is a negative. We symmetrize
    over the query->passage and passage->query directions.
    """
    logits = (q_emb @ p_emb.t()) / temperature  # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_q = F.cross_entropy(logits, targets)
    loss_p = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_q + loss_p)
