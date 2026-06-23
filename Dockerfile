# Lightweight image: runs the offline mock demo, CLI, and API with no heavy ML.
# Install the optional `ml` extras at build time if you want real embeddings.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the lightweight dependency set first for better layer caching.
COPY requirements-min.txt ./
RUN pip install --no-cache-dir -r requirements-min.txt

# Copy the project and install it (editable not needed for runtime).
COPY pyproject.toml README.md ./
COPY orchestra ./orchestra
COPY data ./data
RUN pip install --no-cache-dir .

EXPOSE 8000

# Default: serve the API + web UI. Override CMD to run the CLI, e.g.:
#   docker run --rm orchestra-rag orchestra demo
CMD ["uvicorn", "orchestra.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
