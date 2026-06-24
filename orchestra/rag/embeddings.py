"""Embedders: a guarded sentence-transformers backend plus a deterministic
hashing fallback that requires only numpy.

The hashing embedder produces genuinely useful retrieval embeddings: it maps
each token to a stable bucket via a hash, accumulates an IDF-agnostic weighted
bag-of-words vector, and L2-normalizes it. Documents that share vocabulary land
near each other under cosine similarity, which is exactly what RAG retrieval
needs — and it is fully deterministic and offline.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "STEmbedder",
    "get_embedder",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Protocol every embedder satisfies."""

    @property
    def dimension(self) -> int:
        """Length of the embedding vectors produced."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts into a ``(len(texts), dimension)`` float array."""
        ...


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if n <= 1:
        return list(tokens)
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class HashingEmbedder:
    """Deterministic, dependency-light embedder using the hashing trick.

    Each unigram and bigram is hashed into one of ``dimension`` buckets. The
    bucket sign is also derived from the hash so collisions partially cancel
    rather than always reinforcing. Vectors are L2-normalized so dot product
    equals cosine similarity.
    """

    def __init__(self, dimension: int = 512, use_bigrams: bool = True) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._use_bigrams = use_bigrams

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash(self, token: str) -> tuple[int, float]:
        # MD5 here is a fast, stable, NON-cryptographic hash for the feature-
        # hashing trick — not a security primitive. usedforsecurity=False makes
        # that explicit (and keeps it working under FIPS).
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
        bucket = int.from_bytes(digest[:4], "little") % self._dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        return bucket, sign

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            return vec
        features = list(tokens)
        if self._use_bigrams:
            features.extend(_ngrams(tokens, 2))
        for feature in features:
            bucket, sign = self._hash(feature)
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])


class STEmbedder:
    """sentence-transformers backed embedder (guarded import).

    Loads a model lazily on first use so importing this module never pulls in
    torch. Raises a clear error if the optional dependency is missing.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._dimension = 0

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "STEmbedder requires sentence-transformers. Install it with "
                "`pip install sentence-transformers`, or use the HashingEmbedder "
                "fallback (set OARAG_EMBEDDER=hashing)."
            ) from exc
        model = SentenceTransformer(self._model_name)
        self._model = model
        self._dimension = int(model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_model()
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        assert self._model is not None
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def _ml_embedder() -> Optional["Embedder"]:
    """Return the from-scratch ML bi-encoder embedder if torch + a trained
    checkpoint are both available, else ``None``. Never imports torch unless a
    checkpoint exists, so the zero-dependency path stays clean.
    """
    try:
        from orchestra.ml import has_torch
        from orchestra.ml.adapters import MLEmbedder, checkpoints_present

        if has_torch() and checkpoints_present():
            return MLEmbedder()
    except Exception:  # pragma: no cover - any import/availability failure
        return None
    return None


def get_embedder(
    kind: str = "auto",
    *,
    dimension: int = 512,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Embedder:
    """Construct an embedder.

    Args:
        kind: ``"hashing"``, ``"ml"`` (the from-scratch bi-encoder),
            ``"sentence-transformers"`` (aka ``"st"``), or ``"auto"``.
            ``"auto"`` prefers the trained from-scratch bi-encoder when torch and
            a checkpoint are present, otherwise sentence-transformers if
            importable, and finally falls back to the deterministic hashing
            embedder — so RAG works with zero heavy dependencies.
        dimension: Dimension for the hashing embedder.
        model_name: Model id for the sentence-transformers backend.
    """
    kind = (kind or "auto").lower()
    if kind in {"ml", "mlbiencoder", "biencoder"}:
        from orchestra.ml.adapters import MLEmbedder

        return MLEmbedder()
    if kind in {"st", "sentence-transformers", "sentencetransformers"}:
        return STEmbedder(model_name=model_name)
    if kind == "hashing":
        return HashingEmbedder(dimension=dimension)
    if kind == "auto":
        ml = _ml_embedder()
        if ml is not None:
            return ml
        try:
            import sentence_transformers  # type: ignore  # noqa: F401

            return STEmbedder(model_name=model_name)
        except ImportError:
            return HashingEmbedder(dimension=dimension)
    raise ValueError(f"Unknown embedder kind: {kind!r}")
