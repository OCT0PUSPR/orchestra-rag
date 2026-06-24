"""Retrieval evaluation: recall@k and nDCG@k.

Given a set of held-out queries each with exactly one known relevant passage,
we rank the whole passage pool with the bi-encoder (and optionally rerank the
top candidates with the cross-encoder) and compute:

* **recall@k** — fraction of queries whose relevant passage is in the top-k.
* **nDCG@k** — normalized discounted cumulative gain (single relevant doc, so
  IDCG = 1 and nDCG = 1/log2(rank+1) when the doc is in the top-k, else 0).

These metrics are model-agnostic: :func:`recall_at_k` / :func:`ndcg_at_k` take a
list of ranked ids and the gold id, so the same harness serves both the ML
models and the RAG pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "RetrievalMetrics",
    "evaluate_ranker",
    "evaluate_biencoder",
]


def recall_at_k(ranked_ids: Sequence[int], gold_id: int, k: int) -> float:
    """1.0 if ``gold_id`` is in the top-k, else 0.0 (single relevant doc)."""
    return 1.0 if gold_id in list(ranked_ids)[:k] else 0.0


def ndcg_at_k(ranked_ids: Sequence[int], gold_id: int, k: int) -> float:
    """nDCG@k for a single relevant document (binary relevance)."""
    top = list(ranked_ids)[:k]
    for rank, doc in enumerate(top, start=1):
        if doc == gold_id:
            return 1.0 / math.log2(rank + 1)  # IDCG = 1
    return 0.0


@dataclass
class RetrievalMetrics:
    n: int
    k: int
    recall: float
    ndcg: float
    label: str = ""

    def summary(self) -> str:
        tag = f"[{self.label}] " if self.label else ""
        return f"{tag}n={self.n} recall@{self.k}={self.recall:.4f} nDCG@{self.k}={self.ndcg:.4f}"


def evaluate_ranker(
    rank_fn: Callable[[str, int], List[int]],
    queries: Sequence[str],
    gold_ids: Sequence[int],
    *,
    k: int = 10,
    label: str = "",
) -> RetrievalMetrics:
    """Evaluate any ranker. ``rank_fn(query, k)`` returns ranked passage indices."""
    recalls: List[float] = []
    ndcgs: List[float] = []
    for q, gold in zip(queries, gold_ids):
        ranked = rank_fn(q, k)
        recalls.append(recall_at_k(ranked, gold, k))
        ndcgs.append(ndcg_at_k(ranked, gold, k))
    n = len(queries) or 1
    return RetrievalMetrics(
        n=len(queries),
        k=k,
        recall=sum(recalls) / n,
        ndcg=sum(ndcgs) / n,
        label=label,
    )


def evaluate_biencoder(
    model,  # BiEncoder
    tokenizer,  # BPETokenizer
    queries: Sequence[str],
    gold_ids: Sequence[int],
    passages: Sequence[str],
    *,
    k: int = 10,
    device: str = "cpu",
    reranker=None,  # Optional[CrossEncoder]
    rerank_pool: int = 50,
    label: str = "biencoder",
) -> Tuple[RetrievalMetrics, Optional[RetrievalMetrics]]:
    """Score the bi-encoder alone and (optionally) bi-encoder + cross-encoder.

    Returns ``(bi_metrics, rerank_metrics_or_None)``.
    """
    import torch

    pool_emb = model.encode_texts(list(passages), tokenizer, device=device)
    q_emb = model.encode_texts(list(queries), tokenizer, device=device)
    sims = q_emb @ pool_emb.t()  # (Q, P)
    bi_ranked = torch.argsort(sims, dim=1, descending=True).tolist()

    def bi_rank_fn(qi: int) -> List[int]:
        return bi_ranked[qi]

    recalls, ndcgs = [], []
    for i, gold in enumerate(gold_ids):
        ranked = bi_rank_fn(i)
        recalls.append(recall_at_k(ranked, gold, k))
        ndcgs.append(ndcg_at_k(ranked, gold, k))
    n = len(queries) or 1
    bi_metrics = RetrievalMetrics(
        n=len(queries), k=k, recall=sum(recalls) / n, ndcg=sum(ndcgs) / n, label=label
    )

    if reranker is None:
        return bi_metrics, None

    # Rerank the top ``rerank_pool`` bi-encoder candidates with the cross-encoder.
    rr_recalls, rr_ndcgs = [], []
    for i, gold in enumerate(gold_ids):
        cand_idx = bi_ranked[i][:rerank_pool]
        cand_texts = [passages[j] for j in cand_idx]
        scores = reranker.score_pairs(queries[i], cand_texts, tokenizer, device=device)
        order = sorted(range(len(cand_idx)), key=lambda x: scores[x], reverse=True)
        ranked = [cand_idx[o] for o in order]
        rr_recalls.append(recall_at_k(ranked, gold, k))
        rr_ndcgs.append(ndcg_at_k(ranked, gold, k))
    rerank_metrics = RetrievalMetrics(
        n=len(queries),
        k=k,
        recall=sum(rr_recalls) / n,
        ndcg=sum(rr_ndcgs) / n,
        label=f"{label}+rerank",
    )
    return bi_metrics, rerank_metrics
