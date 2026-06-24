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
from orchestra.rag.hybrid import hybrid_search
from orchestra.rag.loaders import load_paths
from orchestra.rag.rerank import Reranker
from orchestra.rag.sparse import BM25Index
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
        reranker: Optional[Reranker] = None,
    ) -> None:
        # Use ``is None`` (not truthiness): a freshly-built store has __len__ == 0
        # and is therefore *falsy*, so ``store or ...`` would wrongly discard it.
        self.embedder: Embedder = embedder if embedder is not None else get_embedder("auto")
        self.store: VectorStore = store if store is not None else get_vector_store("numpy")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.reranker = reranker
        # Sparse index + text/meta maps power hybrid (BM25 + dense) retrieval.
        self._bm25 = BM25Index()
        self._text_by_id: Dict[str, str] = {}
        self._meta_by_id: Dict[str, Dict[str, object]] = {}

    def _index_sparse(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metas: Sequence[Dict[str, object]],
    ) -> None:
        for doc_id, text, meta in zip(ids, texts, metas):
            if doc_id in self._text_by_id:
                continue  # idempotent: skip already-indexed chunk ids
            self._bm25.add(doc_id, text)
            self._text_by_id[doc_id] = text
            self._meta_by_id[doc_id] = dict(meta)

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
        self._index_sparse(all_ids, all_texts, all_meta)
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
        self._index_sparse(all_ids, all_texts, all_meta)
        self.store.persist()
        return len(all_texts)

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, query: str, k: int = 4, *, hybrid: bool = False) -> List[Passage]:
        """Retrieve the top-``k`` passages for a query, numbered for citation.

        Args:
            query: The user query.
            k: Number of passages to return.
            hybrid: If true, fuse dense + BM25 sparse retrieval (with optional
                cross-encoder reranking). Otherwise use dense-only similarity.
        """
        if not query.strip():
            return []
        if hybrid:
            return self._retrieve_hybrid(query, k)
        query_vec = self.embedder.embed([query])[0]
        hits: List[SearchResult] = self.store.similarity_search(query_vec, k=k)
        return self._to_passages(
            [(h.id, h.text, str(h.metadata.get("source", h.id)), h.score) for h in hits]
        )

    def _retrieve_hybrid(self, query: str, k: int) -> List[Passage]:
        pool = max(k, 20)
        query_vec = self.embedder.embed([query])[0]
        dense_hits = self.store.similarity_search(query_vec, k=pool)
        fused = hybrid_search(
            dense_hits,
            self._bm25,
            query,
            k=k,
            candidate_pool=pool,
            reranker=self.reranker,
            text_by_id=self._text_by_id,
            meta_by_id=self._meta_by_id,
        )
        return self._to_passages(
            [
                (f.doc_id, f.text, str(f.metadata.get("source", f.doc_id)), f.score)
                for f in fused
            ]
        )

    @staticmethod
    def _to_passages(rows: List[tuple]) -> List[Passage]:
        passages: List[Passage] = []
        for i, (chunk_id, text, source, score) in enumerate(rows, start=1):
            passages.append(
                Passage(
                    citation=i,
                    text=text,
                    source=source,
                    score=float(score),
                    chunk_id=chunk_id,
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
