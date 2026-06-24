"""Hardened FastAPI application.

Endpoints
---------
* ``GET  /``              — serves the dark web UI.
* ``GET  /health``        — liveness.
* ``GET  /ready``         — readiness: checks DB + vector store.
* ``GET  /metrics``       — Prometheus exposition.
* ``GET  /collections``   — list the caller's collections (multi-tenant).
* ``POST /collections``   — create a collection.
* ``DELETE /collections/{name}`` — delete a collection.
* ``POST /ingest``        — validated multipart upload into a collection.
* ``POST /ask``           — SSE stream of the multi-agent collaboration.

Security: optional API-key auth (``OARAG_REQUIRE_AUTH``), per-key rate limiting,
CORS allowlist, security headers, strict upload validation, and a global error
handler. Each API key sees only its own collections (tenant isolation). When auth
is disabled (the default for the offline demo) a shared ``public`` tenant is used
so the UI works out of the box.

NOTE: this module intentionally does NOT use ``from __future__ import
annotations`` — FastAPI/pydantic resolve route-handler annotations at runtime,
and stringized forward references (e.g. ``List[UploadFile]``) break that.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from orchestra.app import build_orchestrator, build_pipeline, default_corpus_dir
from orchestra.config import load_settings
from orchestra.observability import (
    M,
    configure_logging,
    get_logger,
    metrics_content_type,
    metrics_text,
    new_request_id,
)
from orchestra.security import RateLimiter, UploadValidationError, validate_upload

_WEB_DIR = Path(__file__).resolve().parent / "web"
_log = get_logger("orchestra.api")


class TenantStore:
    """Holds one isolated RAG pipeline per (tenant, collection).

    This is the in-process isolation boundary: a caller's key only ever touches
    pipelines under its own tenant id, so it cannot read another tenant's KB.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._pipelines: Dict[str, object] = {}

    @staticmethod
    def _key(tenant: str, collection: str) -> str:
        return f"{tenant}::{collection}"

    def get(self, tenant: str, collection: str):
        key = self._key(tenant, collection)
        if key not in self._pipelines:
            self._pipelines[key] = build_pipeline(self._settings)
        return self._pipelines[key]

    def has(self, tenant: str, collection: str) -> bool:
        return self._key(tenant, collection) in self._pipelines

    def collections(self, tenant: str) -> List[str]:
        prefix = f"{tenant}::"
        return sorted(k[len(prefix):] for k in self._pipelines if k.startswith(prefix))

    def delete(self, tenant: str, collection: str) -> bool:
        key = self._key(tenant, collection)
        return self._pipelines.pop(key, None) is not None


def create_app():
    """Construct and return the hardened FastAPI application."""
    from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    settings = load_settings()
    configure_logging(json_logs=True)

    # Optional DB for readiness + run audit. Degrades gracefully if unavailable.
    db = None
    try:
        from orchestra.db import get_database

        db = get_database(settings.database_url)
        db.create_all()
    except Exception as exc:  # pragma: no cover - db optional at runtime
        _log.warning("db_unavailable", error=str(exc))

    store = TenantStore(settings)
    rate_limiter = RateLimiter(per_minute=settings.rate_limit_per_minute)

    # Seed the public tenant's default collection with the bundled corpus so the
    # demo works immediately with no auth.
    public_default = store.get("public", "default")
    if len(public_default) == 0 and default_corpus_dir().exists():  # type: ignore[arg-type]
        public_default.ingest(default_corpus_dir())  # type: ignore[attr-defined]
        M.kb_chunks.labels(tenant="public").set(len(public_default))  # type: ignore[arg-type]

    app = FastAPI(
        title="orchestra-rag",
        description="Multi-agent orchestration grounded by a shared RAG knowledge base.",
        version="0.1.0",
    )

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*", "X-API-Key"],
    )

    # -- middleware: request id + security headers ------------------------
    @app.middleware("http")
    async def add_context(request: Request, call_next):
        rid = new_request_id()
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - handled by exception handler
            raise
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; img-src 'self' data:",
        )
        return response

    # -- global error handler ---------------------------------------------
    @app.exception_handler(Exception)
    async def on_error(request: Request, exc: Exception):  # pragma: no cover - safety net
        M.errors.labels(component="api").inc()
        _log.error("unhandled_error", path=str(request.url.path), error=str(exc))
        return JSONResponse({"detail": "internal server error"}, status_code=500)

    # -- auth dependency ---------------------------------------------------
    def authenticate(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> str:
        """Resolve the caller's tenant id. Returns 'public' when auth is off."""
        tenant: str
        if not settings.require_auth:
            tenant = "public"
        else:
            if not x_api_key:
                raise HTTPException(status_code=401, detail="missing X-API-Key")
            resolved = _resolve_tenant(db, x_api_key)
            if resolved is None:
                raise HTTPException(status_code=401, detail="invalid API key")
            tenant = resolved
        identity = x_api_key or request.client.host if request.client else tenant
        if not rate_limiter.allow(identity):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        return tenant

    class AskRequest(BaseModel):
        question: str
        collection: str = "default"
        backend: Optional[str] = None
        strategy: Optional[str] = None
        k: Optional[int] = None
        hybrid: Optional[bool] = None

    class CreateCollectionRequest(BaseModel):
        name: str

    # -- health / ready / metrics -----------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0", "backend": settings.backend}

    @app.get("/ready")
    def ready():
        checks = {
            "vector_store": True,  # in-process store is always available
            "database": bool(db.healthy()) if db is not None else False,
        }
        ok = checks["vector_store"]  # DB is optional; vector store is required
        status = 200 if ok else 503
        return JSONResponse({"ready": ok, "checks": checks}, status_code=status)

    @app.get("/metrics")
    def metrics():
        return Response(content=metrics_text(), media_type=metrics_content_type())

    # -- collections ------------------------------------------------------
    @app.get("/collections")
    def list_collections(tenant: str = Depends(authenticate)):
        names = store.collections(tenant)
        return {
            "collections": [
                {"name": n, "chunks": len(store.get(tenant, n))} for n in names
            ]
        }

    @app.post("/collections")
    def create_collection(
        req: CreateCollectionRequest, tenant: str = Depends(authenticate)
    ):
        name = req.name.strip()
        if not name or "/" in name or "::" in name:
            raise HTTPException(status_code=400, detail="invalid collection name")
        store.get(tenant, name)  # materializes an empty pipeline
        return {"name": name, "chunks": 0}

    @app.delete("/collections/{name}")
    def delete_collection(name: str, tenant: str = Depends(authenticate)):
        if not store.delete(tenant, name):
            raise HTTPException(status_code=404, detail="collection not found")
        return {"deleted": name}

    # -- ingest (validated upload) ----------------------------------------
    @app.post("/ingest")
    async def ingest_upload(
        files: List[UploadFile] = File(...),
        collection: str = "default",
        tenant: str = Depends(authenticate),
    ):
        if not files:
            raise HTTPException(status_code=400, detail="no files uploaded")
        if len(files) > settings.max_upload_files:
            raise HTTPException(
                status_code=400,
                detail=f"too many files: {len(files)} > {settings.max_upload_files}",
            )
        pipeline = store.get(tenant, collection)
        total = 0
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for upload in files:
                content = await upload.read()
                try:
                    safe = validate_upload(
                        upload.filename or "upload.txt",
                        content,
                        upload.content_type,
                        max_bytes=settings.max_upload_bytes,
                    )
                except UploadValidationError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                (tmp_dir / safe).write_bytes(content)
            total = pipeline.ingest(tmp_dir)
        M.ingest_docs.labels(tenant=tenant).inc(len(files))
        M.ingest_chunks.labels(tenant=tenant).inc(total)
        M.kb_chunks.labels(tenant=tenant).set(len(pipeline))
        return {"ingested_chunks": total, "total_chunks": len(pipeline), "collection": collection}

    # -- ask (SSE) --------------------------------------------------------
    @app.post("/ask")
    def ask(req: AskRequest, tenant: str = Depends(authenticate)):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")
        if not store.has(tenant, req.collection):
            raise HTTPException(status_code=404, detail="collection not found")
        pipeline = store.get(tenant, req.collection)
        if len(pipeline) == 0:
            raise HTTPException(status_code=400, detail="collection is empty; ingest first")

        run_settings = load_settings()
        if req.backend:
            run_settings.backend = req.backend
        if req.strategy:
            run_settings.strategy = req.strategy
        if req.k:
            run_settings.k = req.k
        if req.hybrid is not None:
            run_settings.hybrid = req.hybrid

        orch = build_orchestrator(pipeline, run_settings, backend=run_settings.backend)
        run_id = new_request_id()

        def event_stream():
            last_beat = time.monotonic()
            try:
                for event in orch.stream(req.question):
                    payload: dict = {
                        "type": event.type,
                        "role": event.role,
                        "content": event.content,
                        "round": event.round,
                        "run_id": run_id,
                    }
                    if event.type == "final":
                        result = event.metadata.get("result")
                        payload["content"] = result.answer if result else event.content
                        payload["citations"] = result.citations() if result else []
                        payload["approved"] = bool(event.metadata.get("approved"))
                        payload["rounds"] = result.rounds if result else event.round
                        _persist_run(db, tenant, run_id, req, result)
                    elif event.type == "agent_message":
                        payload["approved"] = bool(event.metadata.get("approved"))
                        payload["num_passages"] = event.metadata.get("num_passages")
                    yield f"data: {json.dumps(payload)}\n\n"
                    # Periodic SSE comment heartbeat keeps proxies from closing.
                    if time.monotonic() - last_beat > 10:
                        last_beat = time.monotonic()
                        yield ": keepalive\n\n"
                yield 'data: {"type": "done"}\n\n'
            except Exception as exc:  # pragma: no cover - surface to UI
                M.errors.labels(component="ask").inc()
                err = json.dumps({"type": "error", "content": str(exc)})
                yield f"data: {err}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- web UI -----------------------------------------------------------
    if _WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

        @app.get("/")
        def index():
            index_path = _WEB_DIR / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return JSONResponse({"detail": "web UI not found"}, status_code=404)

    return app


def _resolve_tenant(db, api_key: str) -> Optional[str]:
    """Look up the tenant slug for an API key (hashed compare). None if invalid."""
    if db is None:
        return None
    try:
        from sqlalchemy import select

        from orchestra.db import ApiKey, Tenant
        from orchestra.security import hash_api_key

        with db.session() as s:
            row = s.scalar(
                select(ApiKey).where(
                    ApiKey.key_hash == hash_api_key(api_key), ApiKey.active.is_(True)
                )
            )
            if row is None:
                return None
            tenant = s.get(Tenant, row.tenant_id)
            return tenant.slug if tenant else None
    except Exception:  # pragma: no cover - db error path
        return None


def _persist_run(db, tenant: str, run_id: str, req, result) -> None:
    """Persist a query run + its events for audit. Best-effort."""
    if db is None or result is None:
        return
    try:
        from sqlalchemy import select

        from orchestra.db import QueryRun, RunEvent, Tenant

        with db.session() as s:
            t = s.scalar(select(Tenant).where(Tenant.slug == tenant))
            if t is None:
                t = Tenant(slug=tenant, name=tenant)
                s.add(t)
                s.flush()
            run = QueryRun(
                run_id=run_id,
                tenant_id=t.id,
                question=req.question,
                answer=result.answer,
                strategy=req.strategy or "linear",
                backend=req.backend or "mock",
                approved=result.approved,
                rounds=result.rounds,
                status="completed",
            )
            s.add(run)
            s.flush()
            for seq, ev in enumerate(result.transcript):
                s.add(
                    RunEvent(
                        run_id=run.id,
                        seq=seq,
                        type=ev.type,
                        role=ev.role,
                        content=ev.content[:4000],
                        round=ev.round,
                    )
                )
    except Exception as exc:  # pragma: no cover - audit must never break a request
        _log.warning("run_persist_failed", error=str(exc))


# Convenience for `uvicorn orchestra.api.server:app`.
try:  # pragma: no cover - only when fastapi is installed
    app = create_app()
except Exception:  # pragma: no cover - fastapi not installed
    app = None
