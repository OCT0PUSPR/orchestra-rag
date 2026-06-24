"""Tests for hybrid (dense + BM25) retrieval and RRF fusion."""

from __future__ import annotations

from pathlib import Path

from orchestra.rag.embeddings import HashingEmbedder
from orchestra.rag.hybrid import reciprocal_rank_fusion
from orchestra.rag.pipeline import RAGPipeline
from orchestra.rag.vectorstore import NumpyStore


def _corpus_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "sample_corpus"


def test_rrf_combines_two_rankings():
    scores = reciprocal_rank_fusion(["a", "b", "c"], ["c", "b", "d"])
    # 'b' and 'c' appear in both lists, so they should outrank singletons.
    assert scores["b"] > scores["a"]
    assert scores["c"] > scores["a"]


def test_hybrid_improves_top1_on_battery_question():
    p = RAGPipeline(embedder=HashingEmbedder(512), store=NumpyStore())
    p.ingest(_corpus_dir())
    q = "How long does the Atlas-7 battery last and how fast can it swap?"
    hybrid = p.retrieve(q, k=1, hybrid=True)
    assert hybrid
    # BM25's high-IDF "battery"/"swap" terms surface the product doc at rank 1.
    assert hybrid[0].short_source == "product_atlas7.md"


def test_hybrid_and_dense_both_numbered():
    p = RAGPipeline(embedder=HashingEmbedder(512), store=NumpyStore())
    p.ingest(_corpus_dir())
    for hybrid in (False, True):
        ps = p.retrieve("vacation days policy", k=3, hybrid=hybrid)
        assert [x.citation for x in ps] == list(range(1, len(ps) + 1))


def test_hybrid_empty_query():
    p = RAGPipeline(embedder=HashingEmbedder(512), store=NumpyStore())
    p.ingest(_corpus_dir())
    assert p.retrieve("", k=3, hybrid=True) == []
