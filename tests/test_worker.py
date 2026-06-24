"""Tests for the ingestion worker's dedup + idempotency logic."""

from __future__ import annotations

from orchestra.rag.embeddings import HashingEmbedder
from orchestra.rag.pipeline import RAGPipeline
from orchestra.rag.vectorstore import NumpyStore
from orchestra.worker import content_hash, ingest_job


def _pipeline() -> RAGPipeline:
    return RAGPipeline(embedder=HashingEmbedder(256), store=NumpyStore())


def test_content_hash_stable():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


def test_ingest_job_indexes(tmp_path):
    (tmp_path / "a.md").write_text("Alpha document about robots.", encoding="utf-8")
    (tmp_path / "b.md").write_text("Beta document about batteries.", encoding="utf-8")
    p = _pipeline()
    seen: dict = {}
    res = ingest_job(p, [str(tmp_path)], seen_hashes=seen)
    assert res["ingested_docs"] == 2
    assert res["chunks"] >= 2
    assert len(seen) == 2


def test_ingest_job_is_idempotent(tmp_path):
    (tmp_path / "a.md").write_text("Same content here.", encoding="utf-8")
    p = _pipeline()
    seen: dict = {}
    first = ingest_job(p, [str(tmp_path)], seen_hashes=seen)
    second = ingest_job(p, [str(tmp_path)], seen_hashes=seen)
    assert first["ingested_docs"] == 1
    assert second["ingested_docs"] == 0
    assert second["skipped_docs"] == 1
