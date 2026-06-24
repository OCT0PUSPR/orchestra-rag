"""Adapters that plug the from-scratch ML models into the existing RAG protocols.

* :class:`MLEmbedder` implements ``orchestra.rag.embeddings.Embedder`` using the
  trained bi-encoder. ``embed(texts) -> (n, dim)`` L2-normalized numpy.
* :class:`MLReranker` implements ``orchestra.rag.rerank.Reranker`` using the
  trained cross-encoder.
* :class:`HNSWStore` implements ``orchestra.rag.vectorstore.VectorStore`` backed
  by the from-scratch :class:`~orchestra.ml.hnsw.HNSWIndex`.

These are imported lazily by ``get_embedder`` / ``get_vector_store`` /
``get_reranker`` so the zero-dependency path never imports torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from orchestra.ml.bi_encoder import BiEncoder
    from orchestra.ml.cross_encoder import CrossEncoder
    from orchestra.ml.hnsw import HNSWIndex
    from orchestra.ml.tokenizer import BPETokenizer

__all__ = ["MLEmbedder", "MLReranker", "HNSWStore", "default_checkpoint_dir", "checkpoints_present"]


def default_checkpoint_dir() -> Path:
    """Where trained ML artifacts live by default."""
    return Path(__file__).resolve().parent / "checkpoints"


def checkpoints_present(ckpt_dir: Optional[str | Path] = None) -> bool:
    """True if a trained bi-encoder + tokenizer are available."""
    d = Path(ckpt_dir) if ckpt_dir else default_checkpoint_dir()
    return (d / "bi_encoder.pt").exists() and (d / "tokenizer.json").exists()


class MLEmbedder:
    """Bi-encoder embedder (lazy torch load) implementing the Embedder protocol."""

    def __init__(
        self,
        ckpt_dir: Optional[str | Path] = None,
        *,
        device: str = "auto",
        max_len: int = 96,
    ) -> None:
        self._dir = Path(ckpt_dir) if ckpt_dir else default_checkpoint_dir()
        self._device = device
        self._max_len = max_len
        self._model: Optional["BiEncoder"] = None
        self._tokenizer: Optional["BPETokenizer"] = None
        self._torch_device: Any = None
        self._dim = 0

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from orchestra.ml.bi_encoder import BiEncoder
        from orchestra.ml.device import select_device
        from orchestra.ml.tokenizer import BPETokenizer

        bi_path = self._dir / "bi_encoder.pt"
        tok_path = self._dir / "tokenizer.json"
        if not bi_path.exists() or not tok_path.exists():
            raise FileNotFoundError(
                f"MLEmbedder needs a trained checkpoint at {self._dir}. "
                "Run `python scripts/train_ml.py` first, or use OARAG_EMBEDDER=hashing."
            )
        dev = select_device(self._device)
        self._torch_device = dev
        self._model = BiEncoder.load(bi_path, map_location=str(dev)).to(dev)
        self._tokenizer = BPETokenizer.load(tok_path)
        self._dim = int(self._model.cfg.dim)

    @property
    def dimension(self) -> int:
        self._ensure()
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure()
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        assert self._model is not None and self._tokenizer is not None
        emb = self._model.encode_texts(
            list(texts),
            self._tokenizer,
            device=str(self._torch_device),
            max_len=self._max_len,
        )
        return emb.numpy().astype(np.float32)


class MLReranker:
    """Cross-encoder reranker (lazy torch load) implementing the Reranker protocol."""

    def __init__(
        self,
        ckpt_dir: Optional[str | Path] = None,
        *,
        device: str = "auto",
    ) -> None:
        self._dir = Path(ckpt_dir) if ckpt_dir else default_checkpoint_dir()
        self._device = device
        self._model: Optional["CrossEncoder"] = None
        self._tokenizer: Optional["BPETokenizer"] = None
        self._torch_device: Any = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from orchestra.ml.cross_encoder import CrossEncoder
        from orchestra.ml.device import select_device
        from orchestra.ml.tokenizer import BPETokenizer

        cross_path = self._dir / "cross_encoder.pt"
        tok_path = self._dir / "tokenizer.json"
        if not cross_path.exists() or not tok_path.exists():
            raise FileNotFoundError(
                f"MLReranker needs a trained checkpoint at {self._dir}. "
                "Run `python scripts/train_ml.py` first, or disable reranking."
            )
        dev = select_device(self._device)
        self._torch_device = dev
        self._model = CrossEncoder.load(cross_path, map_location=str(dev)).to(dev)
        self._tokenizer = BPETokenizer.load(tok_path)

    def rerank(self, query: str, candidates: Sequence[Tuple[str, str]]) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        self._ensure()
        assert self._model is not None and self._tokenizer is not None
        return self._model.rerank(
            query, list(candidates), self._tokenizer, device=str(self._torch_device)
        )


class HNSWStore:
    """A vector store backed by the from-scratch HNSW index (pure numpy)."""

    def __init__(
        self,
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 42,
        persist_dir: Optional[str | Path] = None,
    ) -> None:
        from orchestra.ml.hnsw import HNSWIndex

        self._index_cls = HNSWIndex
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._seed = seed
        self._index: Optional[HNSWIndex] = None
        self._texts: Dict[str, str] = {}
        self._metas: Dict[str, Dict[str, object]] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._index = self._index_cls(
                dim=dim,
                m=self._m,
                ef_construction=self._ef_construction,
                ef_search=self._ef_search,
                seed=self._seed,
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
        emb = np.asarray(embeddings, dtype=np.float32)
        self._ensure_index(emb.shape[1])
        assert self._index is not None
        for i, doc_id in enumerate(ids):
            if doc_id in self._texts:
                continue  # HNSW is append-only; skip duplicates idempotently
            self._index.add(doc_id, emb[i])
            self._texts[doc_id] = texts[i]
            self._metas[doc_id] = dict(metadatas[i])

    def similarity_search(self, query_embedding: np.ndarray, k: int = 4):
        from orchestra.rag.vectorstore import SearchResult

        if self._index is None or len(self._index) == 0:
            return []
        hits = self._index.query(np.asarray(query_embedding, dtype=np.float32), k=k)
        results: List[SearchResult] = []
        for doc_id, score in hits:
            results.append(
                SearchResult(
                    id=doc_id,
                    text=self._texts.get(doc_id, ""),
                    metadata=dict(self._metas.get(doc_id, {})),
                    score=float(score),
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._texts)

    def persist(self) -> None:
        # The in-memory graph is rebuilt on each run from the embeddings; we keep
        # persistence a no-op here to avoid pickling the graph (kept simple).
        return None
