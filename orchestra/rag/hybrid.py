"""Hybrid retrieval: fuse dense (vector) and sparse (BM25) results.

Uses Reciprocal Rank Fusion (RRF), which is robust to differing score scales
between the dense and sparse arms. An optional cross-encoder reranker can
re-order the fused candidates for higher precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from orchestra.rag.rerank import Reranker
from orchestra.rag.sparse import BM25Index
from orchestra.rag.vectorstore import SearchResult

__all__ = ["FusedHit", "reciprocal_rank_fusion", "hybrid_search"]


@dataclass
class FusedHit:
    """A fused candidate with its combined score and source texts."""

    doc_id: str
    text: str
    score: float
    metadata: Dict[str, object]


def reciprocal_rank_fusion(
    dense_ids: List[str],
    sparse_ids: List[str],
    *,
    k_rrf: int = 60,
) -> Dict[str, float]:
    """Compute RRF scores from two ranked id lists.

    Each list contributes ``1 / (k_rrf + rank)`` for each id at ``rank`` (0-based).
    """
    scores: Dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
    return scores


def hybrid_search(
    dense_hits: List[SearchResult],
    sparse_index: BM25Index,
    query: str,
    *,
    k: int = 4,
    candidate_pool: int = 20,
    reranker: Optional[Reranker] = None,
    text_by_id: Optional[Dict[str, str]] = None,
    meta_by_id: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[FusedHit]:
    """Fuse dense and sparse results, optionally rerank, and return the top-``k``.

    Args:
        dense_hits: Results from the vector store (already ordered by similarity).
        sparse_index: A populated BM25 index over the same corpus.
        query: The user query.
        k: Number of final results.
        candidate_pool: How many candidates to consider before reranking.
        reranker: Optional cross-encoder reranker.
        text_by_id: Map of doc_id -> text (for ids only present in the sparse arm).
        meta_by_id: Map of doc_id -> metadata.
    """
    text_by_id = dict(text_by_id or {})
    meta_by_id = dict(meta_by_id or {})

    dense_ids: List[str] = []
    for hit in dense_hits:
        dense_ids.append(hit.id)
        text_by_id.setdefault(hit.id, hit.text)
        meta_by_id.setdefault(hit.id, hit.metadata)

    sparse_hits = sparse_index.search(query, k=candidate_pool)
    sparse_ids = [h.doc_id for h in sparse_hits]

    fused_scores = reciprocal_rank_fusion(dense_ids, sparse_ids)
    ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    ordered = ordered[:candidate_pool]

    if reranker is not None:
        candidates = [
            (doc_id, text_by_id.get(doc_id, ""))
            for doc_id, _ in ordered
            if text_by_id.get(doc_id)
        ]
        reranked = reranker.rerank(query, candidates)
        if reranked:
            ordered = reranked

    results: List[FusedHit] = []
    for doc_id, score in ordered[:k]:
        results.append(
            FusedHit(
                doc_id=doc_id,
                text=text_by_id.get(doc_id, ""),
                score=float(score),
                metadata=meta_by_id.get(doc_id, {}),
            )
        )
    return results
