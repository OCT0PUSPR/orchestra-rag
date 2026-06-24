"""Command-line interface for orchestra-rag.

Usage::

    orchestra ingest data/sample_corpus
    orchestra ask "How long does the Atlas-7 battery last?" --backend mock
    orchestra demo

The ``ask`` command streams the multi-agent collaboration live, then prints the
final cited answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from orchestra.app import build_orchestrator, build_pipeline, default_corpus_dir
from orchestra.config import load_settings
from orchestra.orchestrator import OrchestratorResult

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    _console: Optional[Console] = Console()
except ImportError:  # pragma: no cover - rich is optional but in requirements
    _console = None


_ROLE_STYLES = {
    "planner": "bold cyan",
    "researcher": "bold green",
    "coder": "bold yellow",
    "synthesizer": "bold magenta",
    "critic": "bold red",
    "system": "dim",
}


def _emit(role: str, content: str, *, title: str = "") -> None:
    label = title or role
    if _console is not None:
        style = _ROLE_STYLES.get(role, "white")
        _console.print(Panel(Text(content), title=f"[{style}]{label}[/{style}]", expand=True))
    else:  # plain fallback
        print(f"\n=== {label} ===\n{content}")


def _plain(text: str) -> None:
    if _console is not None:
        _console.print(text)
    else:
        print(text)


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.store:
        settings.store = args.store
    rag = build_pipeline(settings)
    paths: List[Path] = [Path(p) for p in args.paths]
    n = rag.ingest(paths)
    _plain(f"Ingested {n} chunks from {len(paths)} path(s). Store now holds {len(rag)} chunks.")
    return 0


def _prepare_rag(settings, ingest_paths: Optional[List[str]]):
    rag = build_pipeline(settings)
    if ingest_paths:
        rag.ingest([Path(p) for p in ingest_paths])
    elif len(rag) == 0:
        # Auto-ingest the bundled corpus so the demo works out of the box.
        corpus = default_corpus_dir()
        if corpus.exists():
            rag.ingest(corpus)
            _plain(f"[dim]Auto-ingested bundled corpus ({len(rag)} chunks).[/dim]"
                   if _console else f"Auto-ingested bundled corpus ({len(rag)} chunks).")
    return rag


def cmd_ask(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.backend:
        settings.backend = args.backend
    if args.strategy:
        settings.strategy = args.strategy
    if args.k:
        settings.k = args.k
    rag = _prepare_rag(settings, args.ingest)
    if len(rag) == 0:
        _plain("Knowledge base is empty. Run `orchestra ingest <path>` first.")
        return 1

    orch = build_orchestrator(rag, settings, backend=settings.backend)

    _plain(f"\n[bold]Question:[/bold] {args.question}\n" if _console else f"\nQuestion: {args.question}\n")
    final: Optional[OrchestratorResult] = None
    for event in orch.stream(args.question):
        if event.type == "start":
            _plain(f"[dim]strategy={event.metadata.get('strategy')} backend={settings.backend}[/dim]"
                   if _console else f"strategy={event.metadata.get('strategy')} backend={settings.backend}")
        elif event.type == "round":
            _plain(f"\n[dim]--- round {event.round} ---[/dim]" if _console else f"\n--- round {event.round} ---")
        elif event.type == "agent_message":
            extra = " (APPROVED)" if event.metadata.get("approved") else ""
            _emit(event.role, event.content, title=f"{event.role}{extra}")
        elif event.type == "final":
            result_obj = event.metadata.get("result")
            final = result_obj if isinstance(result_obj, OrchestratorResult) else None

    if final is not None:
        _plain("\n[bold green]Final answer:[/bold green]" if _console else "\nFinal answer:")
        _plain(final.answer)
        _plain("\n[bold]Citations:[/bold]" if _console else "\nCitations:")
        for c in final.citations():
            _plain(f"  [{c['n']}] {c['source']} (score={c['score']})")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a canned end-to-end demo offline with the mock backend."""
    settings = load_settings()
    settings.backend = "mock"
    rag = build_pipeline(settings)
    rag.ingest(default_corpus_dir())
    _plain(f"Ingested bundled corpus: {len(rag)} chunks.\n")
    questions = [
        "How long does the Atlas-7 battery last and how fast can it swap?",
        "What programming languages are approved for production at Nimbus?",
        "How much parental leave do employees get?",
    ]
    for q in questions:
        orch = build_orchestrator(rag, settings, backend="mock")
        _plain("\n" + "=" * 70)
        _plain(f"[bold]Q:[/bold] {q}" if _console else f"Q: {q}")
        result = orch.run(q)
        _plain(f"\n[bold green]A:[/bold green] {result.answer}" if _console else f"\nA: {result.answer}")
        _plain("Citations: " + ", ".join(f"[{c['n']}] {c['source']}" for c in result.citations()))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the RAG evaluation harness and print precision/recall/groundedness."""
    from orchestra.eval import evaluate

    settings = load_settings()
    rag = build_pipeline(settings)
    rag.ingest(default_corpus_dir())
    _plain(f"Ingested bundled corpus: {len(rag)} chunks.\n")
    for hybrid in (False, True):
        result = evaluate(rag, k=args.k or 4, hybrid=hybrid)
        _plain(result.summary())
        if args.verbose:
            for row in result.per_question:
                _plain(
                    f"  {row['id']:10s} P={row['precision']:.2f} R={row['recall']:.2f} "
                    f"grounded={row['grounded']} cite_ok={row['citation_integrity']}"
                )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orchestra", description="orchestra-rag CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest documents into the knowledge base")
    p_ingest.add_argument("paths", nargs="+", help="Files or directories to ingest")
    p_ingest.add_argument("--store", help="Vector store backend (numpy|chroma)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question; stream the agent collaboration")
    p_ask.add_argument("question", help="The question to answer")
    p_ask.add_argument("--backend", help="LLM backend (mock|anthropic|huggingface)")
    p_ask.add_argument("--strategy", help="Orchestration strategy (linear|blackboard)")
    p_ask.add_argument("--k", type=int, help="Number of passages to retrieve")
    p_ask.add_argument("--hybrid", action="store_true", help="Use hybrid dense+BM25 retrieval")
    p_ask.add_argument(
        "--ingest",
        nargs="+",
        help="Optionally ingest these paths before asking",
    )
    p_ask.set_defaults(func=cmd_ask)

    p_demo = sub.add_parser("demo", help="Run a canned offline demo (mock backend)")
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("eval", help="Run the RAG evaluation harness")
    p_eval.add_argument("--k", type=int, help="Top-k for retrieval metrics")
    p_eval.add_argument("--verbose", action="store_true", help="Print per-question rows")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
