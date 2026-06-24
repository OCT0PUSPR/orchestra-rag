# Architecture

This document describes how **orchestra-rag** is put together: the retrieval
stack (including the from-scratch ML models), the multi-agent orchestration
layer, and the production-hardening layer (multi-tenant API, persistence,
observability, security).

---

## 1. Layers at a glance

```
┌──────────────────────────────────────────────────────────────────────┐
│  Surfaces:  rich CLI  ·  FastAPI + SSE  ·  dark web UI                  │
├──────────────────────────────────────────────────────────────────────┤
│  Orchestration:  Planner → Researcher → Synthesizer → Critic           │
│                  (linear pipeline  ·  blackboard loop)                  │
├──────────────────────────────────────────────────────────────────────┤
│  RAG pipeline:  loaders → chunker → embedder → vector index → context  │
│                 (+ BM25 sparse arm, RRF hybrid fusion, reranker)       │
├──────────────────────────────────────────────────────────────────────┤
│  Retrieval models (FROM SCRATCH, orchestra/ml):                        │
│     bi-encoder (InfoNCE)  ·  cross-encoder reranker  ·  HNSW  ·  BM25   │
├──────────────────────────────────────────────────────────────────────┤
│  Hardening:  multi-tenant auth · SQLAlchemy/Alembic · metrics/logs     │
│              rate limiting · upload validation · arq worker            │
└──────────────────────────────────────────────────────────────────────┘
```

A core design principle runs through every layer: **the heavy path is optional**.
The whole system — ingest, retrieve, multi-agent answer, full test suite — runs
with only numpy and stdlib (`HashingEmbedder` + `NumpyStore` + pure-Python BM25
+ `MockLLM`). torch, the trained models, and external services are strictly
opt-in and are loaded behind guarded, *lazy* imports.

---

## 2. The from-scratch retrieval models (`orchestra/ml`)

Everything in `orchestra/ml` is implemented from scratch in PyTorch + numpy. We
use the `tokenizers` library for the **BPE algorithm only** (merge-table training
and the byte-level pre-tokenizer); the model, attention, pooling, losses,
indexes, training loops, evaluation, and ONNX export are all hand-written. No
`sentence-transformers`, no `transformers`, no `faiss`/`hnswlib`.

### 2.1 Transformer encoder — `transformer.py`

A small, from-scratch encoder:

- **`MultiHeadSelfAttention`** — explicit Q/K/V projections, scaled dot-product
  attention, additive padding mask (`-inf` on padded keys, NaN-guarded softmax).
- **`FeedForward`** — position-wise GELU MLP.
- **`TransformerBlock`** — pre-LayerNorm residual block (attention + FFN).
- **`TransformerEncoder`** — token + learned-positional + segment (token-type)
  embeddings → N blocks → final LayerNorm. Weights initialized N(0, 0.02).
- **`mean_pool`** — masked mean over non-pad token states.

### 2.2 Bi-encoder — `bi_encoder.py`

A **shared-tower** bi-encoder: query and passage pass through the same encoder,
are mean-pooled and L2-normalized into a unit-sphere embedding. Trained with
**InfoNCE / in-batch negatives** (`info_nce_loss`): for a batch of `(query,
positive)` pairs, passage `i` is the positive for query `i` and every other
passage in the batch is a negative; the loss is symmetrized over the
query→passage and passage→query directions with a temperature of 0.05.

### 2.3 Cross-encoder reranker — `cross_encoder.py`

A joint scorer: it encodes `[CLS] query [SEP] passage [SEP]` (with segment ids)
and projects the `[CLS]` state through a small head to a single relevance logit.
Trained with **binary cross-entropy** on `(query, positive)=1` vs
`(query, hard-negative)=0`, where hard negatives are (a) the gold BM25 negatives
that ship with the MS-MARCO triplets and (b) extra negatives **mined by the
trained bi-encoder** (`data.mine_hard_negatives`). At query time it re-scores the
top bi-encoder candidates for higher precision.

### 2.4 HNSW index — `hnsw.py`

A from-scratch **Hierarchical Navigable Small World** graph (Malkov & Yashunin),
pure numpy + stdlib:

- random level assignment `⌊-ln(U)·mL⌋`;
- insertion descends greedily through upper layers (ef=1) then beam-searches each
  layer at/below the node level (`ef_construction`), connecting to the `M`
  nearest neighbours (`2M` on layer 0) with neighbour-degree pruning;
- search descends greedily then beam-searches layer 0 with `ef` and returns
  top-k by cosine similarity.

It reaches ~1.0 recall@10 vs brute force on a few-thousand-vector pool (see
`tests/test_ml.py::test_hnsw_recall_matches_bruteforce`).

### 2.5 BM25 — `orchestra/rag/sparse.py`

Okapi BM25 over an in-memory inverted index (pure Python, `k1=1.5`, `b=0.75`,
non-negative IDF). It is the sparse arm of hybrid retrieval and is reused by the
ML stack as a lexical baseline.

### 2.6 Data, training, evaluation, export

- **`data.py`** — builds `(query, positive)` pairs from **MS-MARCO BM25 triplets**
  (auto-downloaded as JSON rows via the public Hugging Face datasets-server, no
  auth/`datasets` lib needed; cached under `artifacts/data/`) plus **synthetic**
  pairs derived from the project corpus (so it works fully offline). Also mines
  bi-encoder hard negatives.
- **`train_biencoder.py` / `train_cross.py`** — AdamW, linear-warmup +
  cosine-decay LR, gradient clipping, per-epoch validation, best-checkpoint
  saving. Device auto-selects **MPS > CUDA > CPU** (`device.py`).
- **`eval.py`** — `recall@k` and `nDCG@k` (binary relevance, single gold doc per
  query). `evaluate_biencoder` ranks the full passage pool with the bi-encoder
  and, optionally, reranks the top candidates with the cross-encoder — returning
  both metrics so the effect of reranking is explicit (it is reported honestly in
  the README, including where the tiny reranker does not beat its first stage).
  `scripts/eval_report.py` prints the full BM25 / bi-encoder / hybrid / rerank
  table over the held-out MS-MARCO pool.
- **`onnx_export.py`** — exports the bi-encoder to ONNX (dynamic batch/seq) and
  verifies parity against PyTorch with onnxruntime (max abs diff ~1e-7).

### 2.7 Adapters — `adapters.py`

Thin classes that plug the models into the existing RAG protocols:

| Adapter      | Implements protocol                       | Backed by            |
| ------------ | ----------------------------------------- | -------------------- |
| `MLEmbedder` | `rag.embeddings.Embedder`                 | trained bi-encoder   |
| `MLReranker` | `rag.rerank.Reranker`                     | trained cross-encoder|
| `HNSWStore`  | `rag.vectorstore.VectorStore`             | from-scratch HNSW    |

All three load torch and the checkpoints **lazily** on first use.

---

## 3. RAG pipeline (`orchestra/rag`)

`RAGPipeline` is the single object the agents talk to. `ingest()` loads
(`.txt/.md/.html/.pdf`), chunks (overlapping word windows), embeds, and writes to
the vector store **and** the BM25 index. `retrieve()` does either dense-only
similarity or **hybrid** retrieval: dense + BM25 fused by **Reciprocal Rank
Fusion** (`hybrid.py`), then optionally reranked, returning citation-numbered
`Passage` objects. `build_context()` renders them into a numbered, cited context
block for the LLM.

### Backend selection (default → fallback)

```
embedder:  auto → MLEmbedder (if torch + checkpoint)   → sentence-transformers → HashingEmbedder
store:     auto → HNSWStore  (if torch + checkpoint)    → NumpyStore
reranker:  enabled → MLReranker (if torch + checkpoint) → CrossEncoder (ST)     → None
```

The factories (`get_embedder`, `get_vector_store`, `get_reranker`) and
`app.build_pipeline` resolve these. The zero-dependency triple
(`HashingEmbedder` + `NumpyStore` + BM25) is always available.

---

## 4. Orchestration (`orchestra/orchestrator.py`, `orchestra/agents`)

Each role (`Planner`, `Researcher`, `Coder`, `Synthesizer`, `Critic`) subclasses
`Agent`. The orchestrator runs one of two strategies and emits structured
`Event`s (`start`, `agent_start`, `agent_message`, `round`, `final`) that stream
to the CLI and the web UI over SSE:

- **linear** — Planner → Researcher (queries RAG) → Synthesizer (cited answer) →
  Critic, looping Synthesizer↔Critic until APPROVED or `max_rounds`.
- **blackboard** — every agent reads/writes a shared scratchpad until the Critic
  approves or `max_rounds`.

A per-query cost `Budget` (`reliability.py`) caps token spend.

---

## 5. Hardening layer

- **Multi-tenant API** (`api/server.py`) — `X-API-Key` → tenant resolution,
  per-collection ingestion, SSE `/ask`. Auth is off by default (offline demo),
  on via `OARAG_REQUIRE_AUTH`.
- **Persistence** (`db/`, `alembic/`) — SQLAlchemy 2.0 models (Tenant, ApiKey,
  Collection, Document, Chunk, QueryRun, RunEvent); Alembic migration to head.
  Default SQLite; Postgres/pgvector-ready.
- **Security** (`security.py`) — hashed API keys, sliding-window rate limiter,
  filename sanitization + upload size/count validation.
- **Observability** (`observability.py`) — structured JSON logs with a
  request-id contextvar; Prometheus metrics (guarded — no-ops if absent).
- **Reliability** (`reliability.py`) — retry/backoff, token/cost estimation,
  budget enforcement.
- **Async ingestion** (`worker.py`) — optional arq worker.

---

## 6. Testing, CI, and the import-guard contract

The **invariant**: importing `orchestra` (and running the full test suite on the
`requirements-min.txt` path) must succeed **without torch**. This is enforced by:

- every torch/onnx/tokenizers import living *inside* a function or method in
  `orchestra/ml`, never at a module top level that the core imports;
- `orchestra.ml.__init__` exposing `HAS_TORCH` via `importlib.util.find_spec`
  (no import);
- the RAG factories probing availability before touching the ML adapters;
- `tests/test_ml.py` using `pytest.importorskip("torch")` for model tests while
  always running the torch-free tests (HNSW, metrics, data, fallback wiring).

CI installs only `requirements-min.txt`, lints with ruff, type-checks the core,
and runs the offline suite. Training deps live in `requirements-train.txt` /
`pip install '.[train]'`, and the from-scratch models are exercised separately
(`scripts/train_ml.py`).
