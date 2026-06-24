"""Async ingestion worker (arq + redis), with a synchronous fallback.

The actual ingestion logic — content-hash dedup, idempotent re-indexing, batch
embedding — lives in :func:`ingest_job` and is fully testable without redis. The
arq wiring (``WorkerSettings`` + ``enqueue_ingest``) is guarded so the package
imports and the job runs even when arq/redis are absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from orchestra.observability import M, get_logger
from orchestra.rag.loaders import load_paths
from orchestra.rag.pipeline import RAGPipeline

_log = get_logger("orchestra.worker")

__all__ = ["content_hash", "ingest_job", "enqueue_ingest", "WorkerSettings"]


def content_hash(text: str) -> str:
    """Stable SHA-256 of a document's text, used for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_job(
    pipeline: RAGPipeline,
    paths: List[str],
    *,
    seen_hashes: Optional[Dict[str, str]] = None,
    tenant: str = "public",
) -> Dict[str, object]:
    """Idempotently ingest ``paths`` into ``pipeline``.

    Deduplicates by document content hash: a document whose hash has already
    been ingested is skipped, making re-runs idempotent. Returns a summary.
    """
    seen_hashes = seen_hashes if seen_hashes is not None else {}
    documents = load_paths([Path(p) for p in paths])
    ingested_docs = 0
    skipped_docs = 0
    total_chunks = 0
    to_ingest_texts: List[str] = []
    to_ingest_sources: List[str] = []

    for doc in documents:
        h = content_hash(doc.text)
        if h in seen_hashes:
            skipped_docs += 1
            continue
        seen_hashes[h] = doc.source
        to_ingest_texts.append(doc.text)
        to_ingest_sources.append(doc.source)

    # Batch-embed all new documents in one pass per document via ingest_texts.
    for text, source in zip(to_ingest_texts, to_ingest_sources):
        n = pipeline.ingest_texts([text], source=source)
        total_chunks += n
        ingested_docs += 1

    M.ingest_docs.labels(tenant=tenant).inc(ingested_docs)
    M.ingest_chunks.labels(tenant=tenant).inc(total_chunks)
    _log.info(
        "ingest_job_complete",
        tenant=tenant,
        ingested_docs=ingested_docs,
        skipped_docs=skipped_docs,
        chunks=total_chunks,
    )
    return {
        "ingested_docs": ingested_docs,
        "skipped_docs": skipped_docs,
        "chunks": total_chunks,
        "total_chunks": len(pipeline),
    }


# --------------------------------------------------------------------------- #
# arq wiring (guarded)                                                         #
# --------------------------------------------------------------------------- #


async def _arq_ingest(ctx: dict, paths: List[str], tenant: str = "public") -> Dict[str, object]:
    """arq task entry point. Builds a pipeline lazily inside the worker process."""
    from orchestra.app import build_pipeline

    pipeline = ctx.get("pipeline")
    if pipeline is None:
        pipeline = build_pipeline()
        ctx["pipeline"] = pipeline
    return ingest_job(pipeline, paths, tenant=tenant)


async def enqueue_ingest(paths: List[str], *, redis_url: str = "redis://localhost:6379") -> str:
    """Enqueue an ingestion job onto redis via arq. Returns the job id."""
    try:
        from arq import create_pool  # type: ignore
        from arq.connections import RedisSettings  # type: ignore
    except ImportError as exc:  # pragma: no cover - arq optional
        raise ImportError("enqueue_ingest requires `arq`. Install with `pip install arq`.") from exc
    pool = await create_pool(RedisSettings.from_dsn(redis_url))
    job = await pool.enqueue_job("_arq_ingest", paths)
    return job.job_id if job else ""


class WorkerSettings:
    """arq WorkerSettings. Run with: ``arq orchestra.worker.WorkerSettings``."""

    functions = [_arq_ingest]

    @staticmethod
    def redis_settings():  # pragma: no cover - requires redis
        import os

        from arq.connections import RedisSettings  # type: ignore

        return RedisSettings.from_dsn(os.environ.get("OARAG_REDIS_URL", "redis://localhost:6379"))


def main() -> int:  # pragma: no cover - CLI entry for `python -m orchestra.worker`
    """Run the arq worker (requires arq + redis)."""
    try:
        from arq import run_worker  # type: ignore
    except ImportError:
        print("arq is not installed. Install with `pip install arq` to run the worker.")
        return 1
    run_worker(WorkerSettings)  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
