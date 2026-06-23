"""Application wiring: build a RAG pipeline and an Orchestrator from settings.

Shared by the CLI and the API server so both behave identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from orchestra.config import Settings, load_settings
from orchestra.llm import LLMBackend, get_llm
from orchestra.orchestrator import Orchestrator
from orchestra.rag.embeddings import get_embedder
from orchestra.rag.pipeline import RAGPipeline
from orchestra.rag.vectorstore import get_vector_store

__all__ = ["build_pipeline", "build_llm", "build_orchestrator", "default_corpus_dir"]


def default_corpus_dir() -> Path:
    """Path to the bundled sample corpus."""
    return Path(__file__).resolve().parent.parent / "data" / "sample_corpus"


def build_pipeline(settings: Optional[Settings] = None) -> RAGPipeline:
    """Construct a RAG pipeline from settings (no persistence by default)."""
    settings = settings or load_settings()
    embedder = get_embedder(
        settings.embedder,
        dimension=settings.embedder_dimension,
    )
    persist_dir = None
    if settings.store == "numpy" and settings.storage_dir:
        persist_dir = str(Path(settings.storage_dir) / "numpy")
    store = get_vector_store(
        settings.store,
        persist_dir=persist_dir if settings.store != "numpy" or settings.storage_dir else None,
    )
    return RAGPipeline(
        embedder=embedder,
        store=store,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )


def build_llm(settings: Optional[Settings] = None, backend: Optional[str] = None) -> LLMBackend:
    """Construct the configured LLM backend."""
    settings = settings or load_settings()
    backend = backend or settings.backend
    if backend in {"anthropic", "claude"}:
        return get_llm("anthropic", model=settings.synthesizer_model)
    if backend in {"huggingface", "hf"}:
        return get_llm("huggingface")
    return get_llm("mock")


def build_orchestrator(
    rag: RAGPipeline,
    settings: Optional[Settings] = None,
    *,
    backend: Optional[str] = None,
    llm: Optional[LLMBackend] = None,
) -> Orchestrator:
    """Construct an orchestrator wired to ``rag`` and the configured backend."""
    settings = settings or load_settings()
    llm = llm or build_llm(settings, backend=backend)
    return Orchestrator(
        llm,
        rag,
        strategy=settings.strategy,
        k=settings.k,
        max_rounds=settings.max_rounds,
    )
