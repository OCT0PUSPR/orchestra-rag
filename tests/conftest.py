"""Shared test fixtures. All offline — no API key, no network."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from orchestra.rag.embeddings import HashingEmbedder
from orchestra.rag.pipeline import RAGPipeline
from orchestra.rag.vectorstore import NumpyStore


@pytest.fixture(autouse=True, scope="session")
def _deterministic_offline_env():
    """Pin the offline test suite to the zero-dependency, deterministic backends
    and an isolated storage dir.

    This makes tests independent of whether trained ML checkpoints happen to be
    present on disk (which would otherwise flip ``embedder=auto`` to the torch
    bi-encoder and change dimensions / require torch), and prevents any stray
    persisted ``storage/`` from leaking between runs.
    """
    tmp = tempfile.mkdtemp(prefix="oarag-tests-")
    prev = {
        k: os.environ.get(k)
        for k in ("OARAG_EMBEDDER", "OARAG_STORE", "OARAG_STORAGE_DIR", "OARAG_BACKEND")
    }
    os.environ["OARAG_EMBEDDER"] = "hashing"
    os.environ["OARAG_STORE"] = "numpy"
    os.environ["OARAG_STORAGE_DIR"] = tmp
    os.environ.setdefault("OARAG_BACKEND", "mock")
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


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
