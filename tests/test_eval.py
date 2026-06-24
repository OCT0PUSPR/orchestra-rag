"""Tests for the RAG evaluation harness."""

from __future__ import annotations

from pathlib import Path

from orchestra.eval import evaluate, load_question_set
from orchestra.rag.embeddings import HashingEmbedder
from orchestra.rag.pipeline import RAGPipeline
from orchestra.rag.vectorstore import NumpyStore


def _ingested() -> RAGPipeline:
    p = RAGPipeline(embedder=HashingEmbedder(512), store=NumpyStore())
    p.ingest(Path(__file__).resolve().parent.parent / "data" / "sample_corpus")
    return p


def test_question_set_loads():
    qs = load_question_set()
    assert len(qs) >= 8
    assert all(q.relevant_sources and q.answer_keywords for q in qs)


def test_eval_recall_is_perfect_at_k4():
    p = _ingested()
    r = evaluate(p, k=4, hybrid=True)
    # Every relevant document is found within the top-4.
    assert r.recall_at_k == 1.0
    assert 0.0 <= r.precision_at_k <= 1.0


def test_eval_citation_integrity_perfect():
    p = _ingested()
    r = evaluate(p, k=4, hybrid=True)
    # The mock synthesizer never hallucinates a citation.
    assert r.citation_integrity == 1.0


def test_hybrid_top1_precision_at_least_dense():
    p = _ingested()
    dense = evaluate(p, k=1, hybrid=False)
    hybrid = evaluate(p, k=1, hybrid=True)
    assert hybrid.precision_at_k >= dense.precision_at_k
    assert hybrid.precision_at_k >= 0.9
