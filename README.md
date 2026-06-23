<div align="center">

# ◐ orchestra-rag

### Multi-agent orchestration, grounded by a shared RAG knowledge base.

[![CI](https://github.com/OCT0PUSPR/orchestra-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/OCT0PUSPR/orchestra-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Backends](https://img.shields.io/badge/LLM-Anthropic%20%7C%20HF%20%7C%20Mock-8a63ff.svg)](#llm-backends)
[![Runs offline](https://img.shields.io/badge/demo-100%25%20offline-2ea043.svg)](#quickstart)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

---

**orchestra-rag** is a small, professional, end-to-end system that pairs a real
**retrieval-augmented-generation pipeline** with a team of **specialized agents**
— a Planner, a Researcher, a Coder, a Critic, and a Synthesizer — coordinated by
an **orchestrator**. The agents share one grounded knowledge base, collaborate
under one of two coordination strategies, and produce a final answer with
**inline citations** that link back to the exact source chunks.

Everything runs **fully offline** out of the box: a bundled `MockLLM` and a
deterministic hashing embedder mean the *whole* system — ingest → retrieve →
multi-agent answer — works in tests and the demo with **no API key and no
network**. Swap in Anthropic Claude or a Hugging Face model with one config flag.

> The corpus is a fictional company, **Nimbus Robotics**, so retrieval returns
> real, checkable, grounded answers even with zero external dependencies.

---

## Features

- 🧩 **Real multi-agent orchestration** — Planner → Researcher → Synthesizer →
  Critic with revise loops, *plus* a blackboard/shared-scratchpad strategy.
- 📚 **Real RAG pipeline** — loaders (`.txt/.md/.html/.pdf`) → overlapping
  chunker → embeddings → vector store → cited context builder.
- 🔌 **Interchangeable backends** behind one `LLMBackend` protocol —
  `MockLLM` (offline), Anthropic Claude (primary), Hugging Face (secondary).
- 🧮 **Zero-heavy-deps RAG** — a deterministic `HashingEmbedder` + a numpy
  cosine-similarity store make retrieval genuinely work without `torch`.
  Optional `sentence-transformers` + `chromadb` are guarded imports.
- 🖥️ **Three surfaces** — a `rich` CLI, a FastAPI server with **SSE streaming**,
  and a **dark web UI** that shows the agents collaborating live.
- 🔗 **Citations that link back** — every claim carries an `[n]` marker; clicking
  it in the UI flashes the source chunk.
- ✅ **Offline test suite** — chunking, embedder determinism, store ranking,
  pipeline retrieval, and a full orchestrator run — no API key, no network.

---

## Architecture

### RAG pipeline

```mermaid
flowchart LR
    A[Documents<br/>.txt .md .html .pdf] --> B[Loaders]
    B --> C[Chunker<br/>overlapping windows]
    C --> D[Embedder<br/>Hashing &#124; SentenceTransformers]
    D --> E[(Vector Store<br/>Numpy &#124; Chroma)]
    Q[User query] --> D2[Embed query]
    D2 --> E
    E --> F[Top-k passages]
    F --> G[build_context<br/>numbered + cited]
    G --> H[Grounded context<br/>for the agents]
```

### Multi-agent orchestration graph

```mermaid
flowchart TD
    U[User question] --> O{Orchestrator}
    O -->|1| P[🧭 Planner<br/>decompose into subtasks]
    P --> R[🔎 Researcher<br/>query RAG, gather cited evidence]
    R -->|retrieve k| KB[(Shared RAG<br/>knowledge base)]
    R --> S[🪄 Synthesizer<br/>write cited answer]
    S --> C[🧐 Critic<br/>check grounding & correctness]
    C -->|NEEDS REVISION| S
    C -->|APPROVED| F[✅ Final cited answer]
    Coder[🛠️ Coder<br/>writes code when asked] -.optional.-> S

    subgraph Strategies
      direction LR
      L[linear pipeline]
      B[blackboard loop<br/>shared scratchpad until Critic approves]
    end
```

Two coordination strategies ship in the box:

- **`linear`** — a fixed pipeline that loops Synthesizer ↔ Critic until the
  Critic approves or `max_rounds` is hit.
- **`blackboard`** — every agent reads and writes a shared scratchpad; the loop
  continues until the Critic approves or `max_rounds` is reached.

Both emit structured events that stream to the CLI and the web UI.

---

## Quickstart

No API key required — the default backend is the offline `MockLLM`.

```bash
# 1. Clone and enter
git clone https://github.com/OCT0PUSPR/orchestra-rag.git
cd orchestra-rag

# 2. Lightweight install (no torch / no chromadb) — enough for everything offline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-min.txt
pip install -e .

# 3. Run the canned offline demo (ingest -> multi-agent -> cited answers)
orchestra demo

# 4. Ask your own question (auto-ingests the bundled corpus on first run)
orchestra ask "How long does the Atlas-7 battery last and how fast can it swap?"

# 5. Run the test suite — fully offline
pytest -q
```

Want real embeddings and Claude answers?

```bash
pip install -r requirements.txt          # adds sentence-transformers, chromadb, torch
export ANTHROPIC_API_KEY=sk-ant-...       # never hardcode keys; env vars only
orchestra ask "Explain Conductor's traffic management" --backend anthropic
```

---

## Usage

### CLI

```bash
# Ingest documents (files or directories) into the knowledge base
orchestra ingest data/sample_corpus
orchestra ingest ./my_docs/handbook.pdf ./my_docs/notes.md

# Ask — streams the live agent collaboration, then prints the cited answer
orchestra ask "What languages are approved for production?" --backend mock
orchestra ask "How does Conductor prevent collisions?" --strategy blackboard --k 5

# Canned multi-question offline demo
orchestra demo
```

### Web UI + API server

```bash
uvicorn orchestra.api.server:app --reload --port 8000
# open http://localhost:8000
```

The dark UI lets you upload docs to build the KB, ask a question, and watch the
Planner / Researcher / Synthesizer / Critic collaborate live in a timeline, with
the final answer showing clickable inline citations.

### API endpoints

| Method | Path            | Description                                              |
| ------ | --------------- | ------------------------------------------------------- |
| `GET`  | `/`             | Serves the web UI                                       |
| `GET`  | `/health`       | Liveness + knowledge-base size                          |
| `POST` | `/ingest`       | Upload docs (multipart `files`) to build the KB         |
| `POST` | `/ingest/paths` | Ingest server-side paths (`{"paths": [...]}`)           |
| `POST` | `/ask`          | **SSE stream** of the collaboration + final cited answer |

```bash
# Health
curl localhost:8000/health

# Ask over SSE (streams agent events as they happen)
curl -N -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How much parental leave do employees get?", "backend": "mock"}'
```

### Python

```python
from pathlib import Path
from orchestra.rag.pipeline import RAGPipeline
from orchestra.llm import MockLLM
from orchestra.orchestrator import Orchestrator

rag = RAGPipeline()                       # auto-picks an embedder + numpy store
rag.ingest(Path("data/sample_corpus"))

orch = Orchestrator(MockLLM(), rag, strategy="linear", k=4, max_rounds=3)
result = orch.run("How fast can the Atlas-7 swap its battery?")

print(result.answer)                      # cited answer
for c in result.citations():
    print(c["n"], c["source"], c["text"][:60])
```

### Docker

```bash
docker compose up app                     # API + UI at http://localhost:8000
docker compose run --rm app orchestra demo
docker compose --profile chroma up        # also start a standalone chroma server
```

---

## How to add an agent role

Adding a specialist is three small steps:

1. **Subclass `Agent`** in `orchestra/agents/roles.py` with a role id and a
   system prompt:

   ```python
   from orchestra.agents.base import Agent, AgentResult

   class FactCheckerAgent(Agent):
       role = "factchecker"

       def default_system_prompt(self) -> str:
           return (
               "You are the Fact-Checker. Verify each claim in the DRAFT against "
               "the numbered CONTEXT and flag any unsupported statement."
           )

       def run(self, task, *, scratchpad=None, draft="", passages=None):
           from orchestra.rag.pipeline import RAGPipeline
           context = RAGPipeline.build_context(passages or [])
           content = self.complete(f"QUESTION: {task}\n\nDRAFT:\n{draft}\n\nCONTEXT:\n{context}")
           return AgentResult(role=self.role, content=content, passages=passages or [])
   ```

2. **Register it** in `build_default_agents(...)` (same file) so the
   orchestrator can find it by name.

3. **Wire it into a strategy** in `orchestra/orchestrator.py` — emit
   `agent_start` / `agent_message` events around its `run(...)` call so the CLI
   and web UI render it automatically (the UI even picks a colour by role name;
   add one in `web/style.css` if you like).

The `MockLLM` keys off the `ROLE:` marker in the system prompt — give your new
role a branch in `orchestra/llm.py` if you want offline behaviour for it too.

---

## Configuration

All settings are env-driven (prefix `OARAG_`) via `pydantic-settings`, with a
`.env` file supported. Copy `.env.example` to `.env`. Secrets are read from the
conventional vars and **never hardcoded**.

| Setting                  | Env var                   | Default              | Notes                                  |
| ------------------------ | ------------------------- | -------------------- | -------------------------------------- |
| LLM backend              | `OARAG_BACKEND`           | `mock`               | `mock` / `anthropic` / `huggingface`   |
| Strategy                 | `OARAG_STRATEGY`          | `linear`             | `linear` / `blackboard`                |
| Retrieved passages       | `OARAG_K`                 | `4`                  | top-k                                  |
| Embedder                 | `OARAG_EMBEDDER`          | `auto`               | `auto` / `hashing` / `sentence-transformers` |
| Vector store             | `OARAG_STORE`             | `numpy`              | `numpy` / `chroma`                     |
| Chunk size / overlap     | `OARAG_CHUNK_SIZE` / `…_OVERLAP` | `180` / `40`  | words                                  |
| Max critic rounds        | `OARAG_MAX_ROUNDS`        | `3`                  |                                        |
| Per-role models          | `OARAG_{ROLE}_MODEL`      | Claude 4.x ids       | used by anthropic/hf backends          |
| Anthropic key            | `ANTHROPIC_API_KEY`       | —                    | secret                                 |
| Hugging Face token       | `HF_TOKEN`                | —                    | secret                                 |

---

## Project tree

```
orchestra-rag/
├── orchestra/
│   ├── __init__.py
│   ├── config.py              # pydantic-settings (+ a no-dep fallback)
│   ├── llm.py                 # LLMBackend protocol: MockLLM / Anthropic / HuggingFace
│   ├── orchestrator.py        # linear + blackboard strategies, streamed events
│   ├── app.py                 # wires pipeline + orchestrator from settings
│   ├── cli.py                 # `orchestra ingest|ask|demo`
│   ├── rag/
│   │   ├── loaders.py         # .txt/.md/.html/.pdf + directory ingest
│   │   ├── chunking.py        # pure overlapping chunker (unit-tested)
│   │   ├── embeddings.py      # HashingEmbedder + guarded STEmbedder
│   │   ├── vectorstore.py     # NumpyStore + guarded ChromaStore
│   │   └── pipeline.py        # ingest / retrieve / build_context
│   ├── agents/
│   │   ├── base.py            # Agent (role, prompt, llm, RAG tool)
│   │   └── roles.py           # Planner/Researcher/Coder/Critic/Synthesizer
│   └── api/
│       ├── server.py          # FastAPI: /ingest /ask(SSE) /health /
│       └── web/               # dark UI: index.html + app.js + style.css
├── data/sample_corpus/        # 5 original Nimbus Robotics docs
├── tests/                     # offline: chunking, embeddings, store, pipeline, orchestrator
├── requirements.txt           # full (incl. heavy ML)
├── requirements-min.txt       # lightweight (no torch/chroma)
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .github/workflows/ci.yml   # lint + offline pytest
```

---

## LLM backends

| Backend       | Class           | Needs                | When                                   |
| ------------- | --------------- | -------------------- | -------------------------------------- |
| `mock`        | `MockLLM`       | nothing              | offline tests, demos, CI               |
| `anthropic`   | `AnthropicLLM`  | `ANTHROPIC_API_KEY`  | primary — Claude with adaptive thinking |
| `huggingface` | `HuggingFaceLLM`| `HF_TOKEN`           | secondary — open models via Inference API |

All three implement the same `LLMBackend` protocol, so they are drop-in
interchangeable. The `MockLLM` is deliberately *not* a dumb echo: it reads the
role marker and the retrieved context to produce role-appropriate, genuinely
grounded output, including real `[n]` citations for the Synthesizer.

---

## Roadmap

- [ ] Streaming token-level output from real backends into the UI timeline.
- [ ] Reranking stage (cross-encoder) between retrieval and synthesis.
- [ ] Tool-calling Coder that actually executes generated code in a sandbox.
- [ ] Per-conversation memory so follow-up questions reuse prior context.
- [ ] Pluggable strategy registry (debate, tree-of-agents, router).
- [ ] Evaluation harness scoring grounding/faithfulness on a labelled set.

---

## License

[MIT](LICENSE) © 2026 OCT0PUSPR
