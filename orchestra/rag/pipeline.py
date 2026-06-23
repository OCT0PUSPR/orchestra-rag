"""The RAG pipeline: ingest -> chunk -> embed -> store, and retrieve -> context.

This is the single object the agents talk to. It is deliberately backend-
agnostic: any :class:`Embedder` and :class:`VectorStore` can be swapped in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from orchestra.rag.chunking import chunk_text
from orchestra.rag.embeddings import Embedder, get_embedder
from orchestra.rag.loaders import load_paths
from orchestra.rag.vectorstore import SearchResult, VectorStore, get_vector_store

logger = logging.getLogger("orchestra.rag.pipeline")

__all__ = ["Passage", "RAGPipeline"]


@dataclass
class Passage:
    """A retrieved, citation-numbered passage."""

    citation: int
    text: str
    source: str
    score: float
    chunk_id: str

    @property
    def short_source(self) -> str:
        """Just the file name, for compact citations."""
        return Path(self.source).name if self.source else self.source


class RAGPipeline:
    """End-to-end retrieval-augmented-generation context builder."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        *,
        chunk_size: int = 180,
        overlap: int = 40,
    ) -> None:
        self.embedder: Embedder = embedder or get_embedder("auto")
        self.store: VectorStore = store or get_vector_store("numpy")
        self.chunk_size = chunk_size
        self.overlap = overlap

    # -- ingestion ---------------------------------------------------------
    def ingest(self, paths: Sequence[str | Path] | str | Path) -> int:
        """Load, chunk, embed, and store the documents at ``paths``.

        Accepts a single path or a sequence of paths (files or directories).
        Returns the number of chunks ingested.
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]
        documents = load_paths(list(paths))
        if not documents:
            logger.warning("No documents loaded from %s", paths)
            return 0

        all_ids: List[str] = []
        all_texts: List[str] = []
        all_meta: List[Dict[str, object]] = []
        for doc in documents:
            chunks = chunk_text(
                doc.text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                source=doc.source,
            )
            for chunk in chunks:
                all_ids.append(chunk.id)
                all_texts.append(chunk.text)
                all_meta.append({"source": chunk.source, "index": chunk.index})

        if not all_texts:
            return 0

        embeddings = self.embedder.embed(all_texts)
        self.store.add(all_ids, embeddings, all_texts, all_meta)
        self.store.persist()
        logger.info("Ingested %d chunks from %d documents", len(all_texts), len(documents))
        return len(all_texts)

    def ingest_texts(
        self,
        texts: Sequence[str],
        *,
        source: str = "inline",
    ) -> int:
        """Ingest raw in-memory texts (one document per string)."""
        all_ids: List[str] = []
        all_texts: List[str] = []
        all_meta: List[Dict[str, object]] = []
        for doc_i, text in enumerate(texts):
            doc_source = f"{source}:{doc_i}" if len(texts) > 1 else source
            for chunk in chunk_text(
                text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                source=doc_source,
            ):
                all_ids.append(chunk.id)
                all_texts.append(chunk.text)
                all_meta.append({"source": chunk.source, "index": chunk.index})
        if not all_texts:
            return 0
        embeddings = self.embedder.embed(all_texts)
        self.store.add(all_ids, embeddings, all_texts, all_meta)
        self.store.persist()
        return len(all_texts)

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, query: str, k: int = 4) -> List[Passage]:
        """Retrieve the top-``k`` passages for a query, numbered for citation."""
        if not query.strip():
            return []
        query_vec = self.embedder.embed([query])[0]
        hits: List[SearchResult] = self.store.similarity_search(query_vec, k=k)
        passages: List[Passage] = []
        for i, hit in enumerate(hits, start=1):
            source = str(hit.metadata.get("source", hit.id))
            passages.append(
                Passage(
                    citation=i,
                    text=hit.text,
                    source=source,
                    score=hit.score,
                    chunk_id=hit.id,
                )
            )
        return passages

    @staticmethod
    def build_context(passages: Sequence[Passage]) -> str:
        """Render passages into a citation-numbered context block for the LLM."""
        if not passages:
            return "(no relevant passages were found in the knowledge base)"
        blocks: List[str] = []
        for p in passages:
            blocks.append(f"[{p.citation}] (source: {p.short_source})\n{p.text}")
        return "\n\n".join(blocks)

    def __len__(self) -> int:
        return len(self.store)
