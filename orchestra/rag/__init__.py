"""RAG pipeline: loaders, chunking, embeddings, vector store, and the pipeline glue."""

from __future__ import annotations

from orchestra.rag.chunking import Chunk, chunk_text
from orchestra.rag.embeddings import Embedder, HashingEmbedder, get_embedder
from orchestra.rag.pipeline import Passage, RAGPipeline
from orchestra.rag.vectorstore import NumpyStore, VectorStore, get_vector_store

__all__ = [
    "Chunk",
    "chunk_text",
    "Embedder",
    "HashingEmbedder",
    "get_embedder",
    "Passage",
    "RAGPipeline",
    "NumpyStore",
    "VectorStore",
    "get_vector_store",
]
