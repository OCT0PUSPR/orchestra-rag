"""End-to-end orchestrator tests with the offline MockLLM.

These prove the whole system runs offline: ingest -> retrieve -> multi-agent
collaboration -> cited answer, with no API key and no network.
"""

from __future__ import annotations

from orchestra.llm import MockLLM
from orchestra.orchestrator import Orchestrator
from orchestra.rag.pipeline import RAGPipeline


def _run(ingested_pipeline: RAGPipeline, question: str, strategy: str = "linear"):
    orch = Orchestrator(
        MockLLM(),
        ingested_pipeline,
        strategy=strategy,
        k=4,
        max_rounds=3,
    )
    return orch.run(question)


def test_linear_run_produces_cited_answer(ingested_pipeline: RAGPipeline):
    result = _run(ingested_pipeline, "How long does the Atlas-7 battery last?")
    assert result.answer.strip()
    # The answer must carry at least one inline citation.
    assert "[1]" in result.answer or "[2]" in result.answer
    assert result.passages
    assert result.citations()


def test_linear_run_is_grounded_and_approved(ingested_pipeline: RAGPipeline):
    result = _run(ingested_pipeline, "How much parental leave do employees get?")
    # With grounded, cited passages the mock Critic approves.
    assert result.approved is True
    assert "leave" in result.answer.lower() or "weeks" in result.answer.lower()


def test_blackboard_strategy_runs(ingested_pipeline: RAGPipeline):
    result = _run(
        ingested_pipeline,
        "What programming languages are approved for production?",
        strategy="blackboard",
    )
    assert result.answer.strip()
    assert result.rounds >= 1
    # The transcript records the collaboration.
    roles = {e.role for e in result.transcript if e.type == "agent_message"}
    assert {"planner", "researcher", "synthesizer", "critic"} <= roles


def test_stream_emits_expected_event_types(ingested_pipeline: RAGPipeline):
    orch = Orchestrator(MockLLM(), ingested_pipeline, strategy="linear", k=4)
    types = [e.type for e in orch.stream("How fast is the Atlas-7?")]
    assert types[0] == "start"
    assert "agent_start" in types
    assert "agent_message" in types
    assert types[-1] == "final"


def test_citations_reference_real_sources(ingested_pipeline: RAGPipeline):
    result = _run(ingested_pipeline, "How does Conductor prevent collisions?")
    cites = result.citations()
    assert cites
    # Every citation must carry a source and real text.
    for c in cites:
        assert c["source"]
        assert c["text"]


def test_unknown_question_is_handled_gracefully(ingested_pipeline: RAGPipeline):
    # A question with no good match still returns a result without crashing.
    result = _run(ingested_pipeline, "zzzqqq nonexistent topic about quantum bananas")
    assert isinstance(result.answer, str)
    assert result.answer.strip()
