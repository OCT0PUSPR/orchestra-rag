# Multi-stage, non-root image for orchestra-rag.
# Stage 1 installs deps + the package into a venv; stage 2 is a slim runtime.

# ---- builder ----
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Lightweight deps only (no torch/chroma) for a small, fast image.
COPY requirements-min.txt ./
RUN pip install --no-cache-dir -r requirements-min.txt

COPY pyproject.toml README.md ./
COPY orchestra ./orchestra
COPY data ./data
COPY alembic.ini ./
COPY alembic ./alembic
RUN pip install --no-cache-dir .

# ---- runtime ----
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    OARAG_BACKEND=mock

# Non-root user.
RUN groupadd --system app && useradd --system --gid app --home /app app

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --from=builder /build/orchestra ./orchestra
COPY --from=builder /build/data ./data
COPY --from=builder /build/alembic.ini ./alembic.ini
COPY --from=builder /build/alembic ./alembic

RUN mkdir -p /app/storage && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "orchestra.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
