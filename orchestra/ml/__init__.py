"""From-scratch ML for orchestra-rag.

This package implements the retrieval models and indexes from scratch:

* a Transformer **bi-encoder** (own attention/blocks + mean-pool) trained with
  InfoNCE / in-batch negatives,
* a Transformer **cross-encoder** reranker trained on hard negatives,
* an **HNSW** index and a **BM25** scorer,
* training, evaluation, and ONNX export.

Everything heavy (torch) is imported lazily *inside* functions/classes so that
``import orchestra.ml`` is cheap and the zero-dependency RAG fallback path never
needs torch. ``HAS_TORCH`` reports availability without importing torch.
"""

from __future__ import annotations

import importlib.util

__all__ = ["HAS_TORCH", "has_torch"]


def has_torch() -> bool:
    """True if torch is importable (without importing it).

    Defensive against import machinery that raises (rather than returning None)
    when the package is absent — in that case torch is treated as unavailable.
    """
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


HAS_TORCH = has_torch()
