"""Observability: structured JSON logging + Prometheus metrics.

Both dependencies are guarded so the package still imports and runs if they are
absent (the logger degrades to stdlib ``logging`` and metrics become no-ops).
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Any, Optional

__all__ = [
    "configure_logging",
    "get_logger",
    "new_request_id",
    "request_id_var",
    "metrics_text",
    "metrics_content_type",
    "M",
]

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    return rid


# --------------------------------------------------------------------------- #
# Structured logging                                                           #
# --------------------------------------------------------------------------- #

_configured = False


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure structlog JSON logging if available, else stdlib logging."""
    global _configured
    if _configured:
        return
    _configured = True
    try:
        import structlog  # type: ignore

        def add_request_id(_logger: Any, _name: str, event_dict: dict) -> dict:
            event_dict.setdefault("request_id", request_id_var.get())
            return event_dict

        processors: list = [
            structlog.contextvars.merge_contextvars,
            add_request_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ]
        if json_logs:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    except ImportError:  # pragma: no cover - structlog optional
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def get_logger(name: str = "orchestra"):
    """Return a structlog logger if available, else a stdlib logger."""
    try:
        import structlog  # type: ignore

        return structlog.get_logger(name)
    except ImportError:  # pragma: no cover - structlog optional
        return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Prometheus metrics                                                           #
# --------------------------------------------------------------------------- #


class _NoopMetric:
    """A metric that does nothing — used when prometheus_client is absent."""

    def labels(self, *_a: Any, **_k: Any) -> "_NoopMetric":
        return self

    def inc(self, *_a: Any, **_k: Any) -> None:
        return None

    def observe(self, *_a: Any, **_k: Any) -> None:
        return None

    def set(self, *_a: Any, **_k: Any) -> None:
        return None


class _Metrics:
    """Lazily-initialized Prometheus metrics with a no-op fallback.

    Each metric attribute is typed ``Any`` because at runtime it is either a
    real prometheus metric or a :class:`_NoopMetric`, depending on whether
    prometheus_client is installed.
    """

    ingest_docs: Any
    ingest_chunks: Any
    retrieval_latency: Any
    rerank_latency: Any
    llm_tokens: Any
    llm_cost: Any
    query_rounds: Any
    errors: Any
    kb_chunks: Any

    def __init__(self) -> None:
        self._registry: Optional[Any] = None
        try:
            from prometheus_client import (  # type: ignore
                CollectorRegistry,
                Counter,
                Gauge,
                Histogram,
            )

            self._registry = CollectorRegistry()
            self.ingest_docs = Counter(
                "orchestra_ingest_documents_total",
                "Documents ingested",
                ["tenant"],
                registry=self._registry,
            )
            self.ingest_chunks = Counter(
                "orchestra_ingest_chunks_total",
                "Chunks ingested",
                ["tenant"],
                registry=self._registry,
            )
            self.retrieval_latency = Histogram(
                "orchestra_retrieval_latency_seconds",
                "Retrieval latency",
                ["mode"],
                registry=self._registry,
            )
            self.rerank_latency = Histogram(
                "orchestra_rerank_latency_seconds",
                "Reranking latency",
                registry=self._registry,
            )
            self.llm_tokens = Counter(
                "orchestra_llm_tokens_total",
                "Estimated LLM tokens",
                ["role", "kind"],
                registry=self._registry,
            )
            self.llm_cost = Counter(
                "orchestra_llm_cost_usd_total",
                "Estimated LLM cost (USD)",
                ["backend"],
                registry=self._registry,
            )
            self.query_rounds = Histogram(
                "orchestra_query_rounds",
                "Critic/revision rounds per query",
                registry=self._registry,
            )
            self.errors = Counter(
                "orchestra_errors_total",
                "Errors",
                ["component"],
                registry=self._registry,
            )
            self.kb_chunks = Gauge(
                "orchestra_kb_chunks",
                "Chunks currently indexed",
                ["tenant"],
                registry=self._registry,
            )
        except ImportError:  # pragma: no cover - prometheus optional
            noop = _NoopMetric()
            self.ingest_docs = noop
            self.ingest_chunks = noop
            self.retrieval_latency = noop
            self.rerank_latency = noop
            self.llm_tokens = noop
            self.llm_cost = noop
            self.query_rounds = noop
            self.errors = noop
            self.kb_chunks = noop

    def render(self) -> bytes:
        if self._registry is None:
            return b"# prometheus_client not installed\n"
        from prometheus_client import generate_latest  # type: ignore

        return generate_latest(self._registry)


M = _Metrics()


def metrics_text() -> bytes:
    """Render the current Prometheus metrics."""
    return M.render()


def metrics_content_type() -> str:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST  # type: ignore

        return CONTENT_TYPE_LATEST
    except ImportError:  # pragma: no cover
        return "text/plain; charset=utf-8"
