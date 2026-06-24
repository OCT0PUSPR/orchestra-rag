"""Tests for the RAG pipeline against the bundled sample corpus."""

from __future__ import annotations

from orchestra.rag.embeddings import HashingEmbedder
from orchestra.rag.pipeline import RAGPipeline
from orchestra.rag.vectorstore import NumpyStore


def test_ingest_loads_chunks(ingested_pipeline: RAGPipeline):
    assert len(ingested_pipeline) > 0


def test_explicit_empty_store_is_not_replaced():
    """Regression: a freshly-built store has len()==0 (falsy); the pipeline must
    keep the *provided* store instead of silently swapping in the default. This
    guards the ``store if store is not None`` fix (an empty HNSW/Numpy store was
    being discarded by the old ``store or get_vector_store(...)``).
    """
    sentinel = NumpyStore()
    pipe = RAGPipeline(embedder=HashingEmbedder(dimension=64), store=sentinel)
    assert pipe.store is sentinel


def test_retrieve_battery_question(ingested_pipeline: RAGPipeline):
    passages = ingested_pipeline.retrieve(
        "How long does the Atlas-7 battery last?", k=4
    )
    assert passages
    joined = " ".join(p.text.lower() for p in passages)
    assert "battery" in joined
    # The product doc should be the top source.
    assert any("atlas7" in p.short_source.lower() or "atlas-7" in p.text.lower() for p in passages)


def test_retrieve_languages_question(ingested_pipeline: RAGPipeline):
    passages = ingested_pipeline.retrieve(
        "What programming languages are approved for production?", k=4
    )
    top_text = " ".join(p.text.lower() for p in passages[:2])
    assert "rust" in top_text or "python" in top_text


def test_retrieve_hr_question(ingested_pipeline: RAGPipeline):
    passages = ingested_pipeline.retrieve(
        "How much parental leave do employees get?", k=4
    )
    joined = " ".join(p.text.lower() for p in passages)
    assert "parental" in joined or "leave" in joined
    assert any("hr" in p.short_source.lower() for p in passages)


def test_citations_are_numbered(ingested_pipeline: RAGPipeline):
    passages = ingested_pipeline.retrieve("vacation days", k=3)
    assert [p.citation for p in passages] == list(range(1, len(passages) + 1))


def test_build_context_includes_citation_markers(ingested_pipeline: RAGPipeline):
    passages = ingested_pipeline.retrieve("traffic management collisions", k=3)
    context = RAGPipeline.build_context(passages)
    assert "[1]" in context
    assert "source:" in context


def test_empty_query_returns_nothing(ingested_pipeline: RAGPipeline):
    assert ingested_pipeline.retrieve("", k=4) == []


def test_ingest_texts(pipeline: RAGPipeline):
    n = pipeline.ingest_texts(
        ["The capital of Nimbusland is Portland and the river is wide."],
        source="inline-doc",
    )
    assert n >= 1
    passages = pipeline.retrieve("what is the capital", k=1)
    assert passages
    assert "portland" in passages[0].text.lower()
