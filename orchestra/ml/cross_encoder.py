"""A cross-encoder reranker: a from-scratch Transformer that jointly encodes
``[CLS] query [SEP] passage [SEP]`` and scores their relevance with a scalar head
on the pooled [CLS] representation.

Trained with binary cross-entropy on (query, positive) vs (query, hard-negative)
pairs, where hard negatives are mined by the trained bi-encoder.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from orchestra.ml.tokenizer import BPETokenizer
from orchestra.ml.transformer import EncoderConfig, TransformerEncoder

__all__ = ["CrossEncoder"]


class CrossEncoder(nn.Module):
    """Joint query-passage scorer. Outputs a single relevance logit per pair."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerEncoder(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask, segment_ids)
        cls = hidden[:, 0]  # [CLS] pooling
        return self.head(cls).squeeze(-1)  # (B,)

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def score_pairs(
        self,
        query: str,
        passages: Sequence[str],
        tokenizer: BPETokenizer,
        *,
        device: "torch.device | str" = "cpu",
        max_len: int = 192,
        batch_size: int = 32,
    ) -> List[float]:
        """Score ``(query, passage_i)`` pairs; returns a relevance score per passage."""
        self.eval()
        scores: List[float] = []
        for start in range(0, len(passages), batch_size):
            chunk = list(passages[start : start + batch_size])
            rows = [tokenizer.encode_pair(query, p, max_len=max_len) for p in chunk]
            ids, mask, seg = tokenizer.pad_pairs(rows)
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            mask_t = torch.tensor(mask, dtype=torch.long, device=device)
            seg_t = torch.tensor(seg, dtype=torch.long, device=device)
            logits = self.forward(ids_t, mask_t, seg_t)
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
        return scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[Tuple[str, str]],
        tokenizer: BPETokenizer,
        *,
        device: "torch.device | str" = "cpu",
    ) -> List[Tuple[str, float]]:
        """Rerank ``(doc_id, text)`` candidates; returns ``(doc_id, score)`` desc."""
        if not candidates:
            return []
        texts = [t for _, t in candidates]
        scores = self.score_pairs(query, texts, tokenizer, device=device)
        scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict()}, str(path))

    @classmethod
    def load(cls, path: str | Path, *, map_location: "str | torch.device" = "cpu") -> "CrossEncoder":
        # First-party checkpoint written by save(); weights_only=False is needed
        # to restore the EncoderConfig dataclass. Never untrusted input. nosec B614.
        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)  # nosec B614
        cfg = EncoderConfig(**ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
