# orchestra-rag — developer + ops tasks.
.PHONY: help install install-min install-train lint format typecheck test cov \
        security audit migrate eval train-ml train-ml-scale demo serve worker \
        docker compose clean precommit

PY ?= python3

help:
	@echo "Targets:"
	@echo "  install        Full install (incl. heavy ML extras + dev tools)"
	@echo "  install-min    Lightweight install (no torch/chroma)"
	@echo "  install-train  From-scratch ML training stack (torch/tokenizers/onnx)"
	@echo "  lint           ruff check"
	@echo "  format         ruff format"
	@echo "  typecheck      mypy"
	@echo "  test           pytest"
	@echo "  cov            pytest with coverage (fails under 80%)"
	@echo "  security       bandit static security scan"
	@echo "  audit          pip-audit dependency vulnerability scan"
	@echo "  migrate        alembic upgrade head"
	@echo "  eval           run the RAG evaluation harness"
	@echo "  train-ml       train the from-scratch bi-/cross-encoders (laptop)"
	@echo "  train-ml-scale train the from-scratch models at scale (GPU)"
	@echo "  demo           run the offline multi-agent demo"
	@echo "  serve          run the API + web UI (uvicorn)"
	@echo "  worker         run the arq ingestion worker"
	@echo "  docker         build the docker image"
	@echo "  compose        docker compose up the full stack"

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e '.[dev]'

install-min:
	$(PY) -m pip install -r requirements-min.txt && $(PY) -m pip install -e .

install-train:
	$(PY) -m pip install -r requirements-min.txt -r requirements-train.txt && $(PY) -m pip install -e '.[dev]'

lint:
	$(PY) -m ruff check orchestra tests

format:
	$(PY) -m ruff format orchestra tests

typecheck:
	$(PY) -m mypy orchestra

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest --cov=orchestra --cov-report=term-missing --cov-fail-under=80 -q

security:
	$(PY) -m bandit -r orchestra -ll -q

audit:
	$(PY) -m pip_audit -r requirements-min.txt || true

migrate:
	$(PY) -m alembic upgrade head

eval:
	$(PY) -m orchestra.cli eval --verbose

train-ml:
	$(PY) scripts/train_ml.py

train-ml-scale:
	$(PY) scripts/scale_up_colab.py

demo:
	$(PY) -m orchestra.cli demo

serve:
	$(PY) -m uvicorn orchestra.api.server:app --host 0.0.0.0 --port 8000 --reload

worker:
	$(PY) -m orchestra.worker

docker:
	docker build -t orchestra-rag:latest .

compose:
	docker compose up --build

precommit:
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage storage *.sqlite
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
