"""Shared test fixtures. All offline — no API key, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestra.rag.embeddings import HashingEmbedder
from orchestra.rag.pipeline import RAGPipeline
from orchestra.rag.vectorstore import NumpyStore


def corpus_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "sample_corpus"


@pytest.fixture()
def pipeline() -> RAGPipeline:
    """A RAG pipeline with the deterministic hashing embedder + numpy store."""
    return RAGPipeline(embedder=HashingEmbedder(dimension=512), store=NumpyStore())


@pytest.fixture()
def ingested_pipeline(pipeline: RAGPipeline) -> RAGPipeline:
    """A pipeline pre-loaded with the bundled sample corpus."""
    n = pipeline.ingest(corpus_dir())
    assert n > 0
    return pipeline
