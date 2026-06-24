"""Optional cross-encoder reranking (guarded import).

If ``sentence-transformers`` is installed, :class:`CrossEncoderReranker` re-scores
candidate passages against the query with a cross-encoder model for higher
precision. When the dependency is absent, :func:`get_reranker` returns ``None``
and the pipeline simply skips reranking — no hard dependency.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

__all__ = ["Reranker", "CrossEncoderReranker", "get_reranker"]


@runtime_checkable
class Reranker(Protocol):
    """Re-score ``(doc_id, text)`` candidates against a query."""

    def rerank(self, query: str, candidates: Sequence[Tuple[str, str]]) -> List[Tuple[str, float]]:
        ...


class CrossEncoderReranker:
    """Cross-encoder reranker backed by sentence-transformers (guarded, lazy)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. Install it "
                "with `pip install sentence-transformers`, or disable reranking."
            ) from exc
        self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, candidates: Sequence[Tuple[str, str]]) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        self._ensure_model()
        assert self._model is not None
        pairs = [[query, text] for _doc_id, text in candidates]
        scores = self._model.predict(pairs)
        scored = [
            (candidates[i][0], float(scores[i])) for i in range(len(candidates))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def get_reranker(enabled: bool = False, model_name: Optional[str] = None) -> Optional[Reranker]:
    """Return a reranker if ``enabled`` and a backend is available, else ``None``.

    Preference order: the from-scratch ML cross-encoder (when torch + a trained
    checkpoint are present), then a sentence-transformers cross-encoder, else
    ``None`` so the pipeline simply skips reranking.
    """
    if not enabled:
        return None
    # 1. Prefer the from-scratch trained cross-encoder.
    try:
        from orchestra.ml import has_torch
        from orchestra.ml.adapters import MLReranker, default_checkpoint_dir

        if has_torch() and (default_checkpoint_dir() / "cross_encoder.pt").exists():
            return MLReranker()
    except Exception:  # pragma: no cover - availability failure
        pass
    # 2. Fall back to a sentence-transformers cross-encoder if installed.
    try:
        import sentence_transformers  # type: ignore  # noqa: F401
    except ImportError:
        return None
    return CrossEncoderReranker(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
