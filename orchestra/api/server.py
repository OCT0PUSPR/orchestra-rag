"""FastAPI application: ingest, multi-agent ask over SSE, health, and web UI.

Endpoints
---------
* ``GET  /``         — serves the dark web UI (``web/index.html``).
* ``GET  /health``   — liveness + knowledge-base size.
* ``POST /ingest``   — upload documents (multipart) or paths (JSON) to build the KB.
* ``POST /ask``      — SSE stream of the multi-agent collaboration + final answer.

The app holds a single shared :class:`RAGPipeline` so ingested documents persist
across requests for the lifetime of the process. On startup it auto-ingests the
bundled sample corpus if the knowledge base is empty, so the UI is useful
immediately.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Optional

from orchestra.app import build_orchestrator, build_pipeline, default_corpus_dir
from orchestra.config import load_settings

_WEB_DIR = Path(__file__).resolve().parent / "web"


def create_app():
    """Construct and return the FastAPI application."""
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    settings = load_settings()
    rag = build_pipeline(settings)

    # Auto-ingest the bundled corpus so the demo is useful immediately.
    if len(rag) == 0:
        corpus = default_corpus_dir()
        if corpus.exists():
            rag.ingest(corpus)

    app = FastAPI(
        title="orchestra-rag",
        description="Multi-agent orchestration grounded by a shared RAG knowledge base.",
        version="0.1.0",
    )

    class AskRequest(BaseModel):
        question: str
        backend: Optional[str] = None
        strategy: Optional[str] = None
        k: Optional[int] = None

    class IngestPathsRequest(BaseModel):
        paths: List[str]

    # -- health -----------------------------------------------------------
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": "0.1.0",
            "chunks": len(rag),
            "backend": settings.backend,
            "strategy": settings.strategy,
        }

    # -- ingest (paths) ---------------------------------------------------
    @app.post("/ingest/paths")
    def ingest_paths(req: IngestPathsRequest):
        n = rag.ingest([Path(p) for p in req.paths])
        return {"ingested_chunks": n, "total_chunks": len(rag)}

    # -- ingest (file upload) --------------------------------------------
    @app.post("/ingest")
    async def ingest_upload(files: List[UploadFile] = File(...)):
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        total = 0
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for upload in files:
                name = Path(upload.filename or "upload.txt").name
                dest = tmp_dir / name
                dest.write_bytes(await upload.read())
            total = rag.ingest(tmp_dir)
        return {"ingested_chunks": total, "total_chunks": len(rag)}

    # -- ask (SSE) --------------------------------------------------------
    @app.post("/ask")
    def ask(req: AskRequest):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")
        if len(rag) == 0:
            raise HTTPException(status_code=400, detail="Knowledge base is empty; ingest first")

        run_settings = load_settings()
        if req.backend:
            run_settings.backend = req.backend
        if req.strategy:
            run_settings.strategy = req.strategy
        if req.k:
            run_settings.k = req.k

        orch = build_orchestrator(rag, run_settings, backend=run_settings.backend)

        def event_stream():
            try:
                for event in orch.stream(req.question):
                    payload: dict = {
                        "type": event.type,
                        "role": event.role,
                        "content": event.content,
                        "round": event.round,
                    }
                    if event.type == "final":
                        result = event.metadata.get("result")
                        payload["content"] = result.answer if result else event.content
                        payload["citations"] = result.citations() if result else []
                        payload["approved"] = bool(event.metadata.get("approved"))
                        payload["rounds"] = result.rounds if result else event.round
                    elif event.type == "agent_message":
                        payload["approved"] = bool(event.metadata.get("approved"))
                        payload["num_passages"] = event.metadata.get("num_passages")
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: {\"type\": \"done\"}\n\n"
            except Exception as exc:  # pragma: no cover - surface errors to UI
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


# Convenience for `uvicorn orchestra.api.server:app`.
try:  # pragma: no cover - only when fastapi is installed
    app = create_app()
except Exception:  # pragma: no cover - fastapi not installed
    app = None
