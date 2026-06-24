#!/usr/bin/env python3
"""Verify the from-scratch ML stack is wired into the RAG pipeline end to end.

Builds a pipeline through the public factories (which should auto-select the
trained bi-encoder + HNSW index + cross-encoder reranker when checkpoints are
present), ingests the bundled corpus, and runs a real retrieval.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestra.app import _resolve_store_kind, build_pipeline, default_corpus_dir  # noqa: E402
from orchestra.config import load_settings  # noqa: E402
from orchestra.rag.embeddings import get_embedder  # noqa: E402
from orchestra.rag.rerank import get_reranker  # noqa: E402


def main() -> int:
    settings = load_settings()
    settings.embedder = "auto"
    settings.store = "auto"
    settings.rerank = True
    settings.hybrid = True

    emb = get_embedder("auto")
    store_kind = _resolve_store_kind("auto")
    reranker = get_reranker(enabled=True)
    print(f"resolved embedder = {type(emb).__name__}")
    print(f"resolved store    = {store_kind}")
    print(f"resolved reranker = {type(reranker).__name__ if reranker else None}")

    rag = build_pipeline(settings)
    n = rag.ingest(default_corpus_dir())
    print(f"ingested {n} chunks; store type = {type(rag.store).__name__}; len = {len(rag)}")

    for q in [
        "How fast can the Atlas-7 swap its battery?",
        "What programming languages are approved for production?",
        "How much parental leave do employees get?",
    ]:
        passages = rag.retrieve(q, k=3, hybrid=True)
        print(f"\nQ: {q}")
        for p in passages:
            print(f"  [{p.citation}] {p.short_source} score={p.score:.3f} :: {p.text[:70].strip()}...")
    print("\nintegration OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
