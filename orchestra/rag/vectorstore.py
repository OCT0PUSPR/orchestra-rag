"""Vector stores: an always-available numpy cosine-similarity store and a
guarded chromadb store. Both satisfy the :class:`VectorStore` protocol.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

logger = logging.getLogger("orchestra.rag.vectorstore")

__all__ = [
    "StoredItem",
    "SearchResult",
    "VectorStore",
    "NumpyStore",
    "ChromaStore",
    "get_vector_store",
]


@dataclass
class StoredItem:
    """A single stored record."""

    id: str
    text: str
    metadata: Dict[str, object]


@dataclass
class SearchResult:
    """A retrieval hit: the stored item plus its similarity score."""

    id: str
    text: str
    metadata: Dict[str, object]
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector stores."""

    def add(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, object]],
    ) -> None:
        """Insert (or upsert) records."""
        ...

    def similarity_search(self, query_embedding: np.ndarray, k: int = 4) -> List[SearchResult]:
        """Return the ``k`` most similar records to ``query_embedding``."""
        ...

    def __len__(self) -> int:
        ...

    def persist(self) -> None:
        """Flush state to disk (no-op for ephemeral stores)."""
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class NumpyStore:
    """In-memory cosine-similarity store with optional numpy-on-disk persistence.

    Embeddings are L2-normalized on insert so a similarity search is a single
    matrix-vector dot product. Persistence writes the matrix as ``.npy`` and the
    text/metadata as a sidecar JSON file.
    """

    def __init__(self, persist_dir: Optional[str | Path] = None) -> None:
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, object]] = []
        self._matrix: Optional[np.ndarray] = None
        self._id_to_row: Dict[str, int] = {}
        if self._persist_dir is not None:
            self._load()

    # -- persistence paths -------------------------------------------------
    def _matrix_path(self) -> Path:
        assert self._persist_dir is not None
        return self._persist_dir / "embeddings.npy"

    def _meta_path(self) -> Path:
        assert self._persist_dir is not None
        return self._persist_dir / "store.json"

    def _load(self) -> None:
        assert self._persist_dir is not None
        meta_path = self._meta_path()
        matrix_path = self._matrix_path()
        if not meta_path.exists() or not matrix_path.exists():
            return
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            self._ids = list(payload["ids"])
            self._texts = list(payload["texts"])
            self._metadatas = list(payload["metadatas"])
            self._matrix = np.load(matrix_path)
            self._id_to_row = {i: r for r, i in enumerate(self._ids)}
        except Exception as exc:  # pragma: no cover - corrupt store
            logger.warning("Failed to load NumpyStore from %s: %s", self._persist_dir, exc)
            self._ids, self._texts, self._metadatas = [], [], []
            self._matrix = None
            self._id_to_row = {}

    # -- core API ----------------------------------------------------------
    def add(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, object]],
    ) -> None:
        if not (len(ids) == len(texts) == len(metadatas) == embeddings.shape[0]):
            raise ValueError("ids, embeddings, texts, and metadatas must align in length")
        if len(ids) == 0:
            return
        normalized = _l2_normalize(np.asarray(embeddings, dtype=np.float32))

        for offset, item_id in enumerate(ids):
            row_vec = normalized[offset : offset + 1]
            if item_id in self._id_to_row:  # upsert
                row = self._id_to_row[item_id]
                assert self._matrix is not None
                self._matrix[row] = row_vec[0]
                self._texts[row] = texts[offset]
                self._metadatas[row] = dict(metadatas[offset])
            else:
                self._id_to_row[item_id] = len(self._ids)
                self._ids.append(item_id)
                self._texts.append(texts[offset])
                self._metadatas.append(dict(metadatas[offset]))
                if self._matrix is None:
                    self._matrix = row_vec.copy()
                else:
                    self._matrix = np.vstack([self._matrix, row_vec])

    def similarity_search(self, query_embedding: np.ndarray, k: int = 4) -> List[SearchResult]:
        if self._matrix is None or len(self._ids) == 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm
        scores = self._matrix @ query  # cosine similarity (both normalized)
        k = max(1, min(k, len(self._ids)))
        # argpartition for the top-k, then sort that slice descending.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results: List[SearchResult] = []
        for row in top_idx:
            results.append(
                SearchResult(
                    id=self._ids[row],
                    text=self._texts[row],
                    metadata=dict(self._metadatas[row]),
                    score=float(scores[row]),
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._ids)

    def persist(self) -> None:
        if self._persist_dir is None:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        matrix = self._matrix if self._matrix is not None else np.zeros((0, 0), dtype=np.float32)
        np.save(self._matrix_path(), matrix)
        self._meta_path().write_text(
            json.dumps(
                {
                    "ids": self._ids,
                    "texts": self._texts,
                    "metadatas": self._metadatas,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


class ChromaStore:
    """chromadb-backed store (guarded import).

    Mirrors the :class:`VectorStore` protocol over a persistent chromadb
    collection. Raises a clear error if chromadb is not installed.
    """

    def __init__(
        self,
        persist_dir: Optional[str | Path] = None,
        collection_name: str = "orchestra_rag",
    ) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ChromaStore requires chromadb. Install it with `pip install chromadb`, "
                "or use NumpyStore (set OARAG_STORE=numpy)."
            ) from exc
        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, object]],
    ) -> None:
        if len(ids) == 0:
            return
        self._collection.upsert(
            ids=list(ids),
            embeddings=[v.tolist() for v in np.asarray(embeddings, dtype=np.float32)],
            documents=list(texts),
            metadatas=[dict(m) for m in metadatas],
        )

    def similarity_search(self, query_embedding: np.ndarray, k: int = 4) -> List[SearchResult]:
        if len(self) == 0:
            return []
        k = max(1, min(k, len(self)))
        res = self._collection.query(
            query_embeddings=[np.asarray(query_embedding, dtype=np.float32).reshape(-1).tolist()],
            n_results=k,
        )
        results: List[SearchResult] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            distance = float(dists[i]) if dists else 0.0
            results.append(
                SearchResult(
                    id=doc_id,
                    text=docs[i] if docs else "",
                    metadata=dict(metas[i]) if metas and metas[i] else {},
                    score=1.0 - distance,  # cosine distance -> similarity
                )
            )
        return results

    def __len__(self) -> int:
        return int(self._collection.count())

    def persist(self) -> None:
        # PersistentClient writes through automatically; nothing to flush.
        return None


def get_vector_store(
    kind: str = "numpy",
    *,
    persist_dir: Optional[str | Path] = None,
    collection_name: str = "orchestra_rag",
) -> VectorStore:
    """Construct a vector store.

    Args:
        kind: ``"numpy"`` (default, always available) or ``"chroma"``.
        persist_dir: Directory for on-disk persistence.
        collection_name: Collection name (chroma only).
    """
    kind = (kind or "numpy").lower()
    if kind == "numpy":
        return NumpyStore(persist_dir=persist_dir)
    if kind in {"chroma", "chromadb"}:
        return ChromaStore(persist_dir=persist_dir, collection_name=collection_name)
    raise ValueError(f"Unknown vector store kind: {kind!r}")
