"""The orchestrator: coordinates agent roles under one of two strategies.

Strategies
----------
* **linear** — a fixed pipeline:
  Planner -> Researcher -> Synthesizer -> Critic -> (revise via Synthesizer) ...
  The Critic can request revisions up to ``max_rounds`` times.

* **blackboard** — a shared-scratchpad loop. Every agent reads and writes a
  shared blackboard; the loop continues until the Critic approves or
  ``max_rounds`` is reached.

Both strategies emit structured :class:`Event` objects (yielded by ``stream``)
so a CLI or web UI can render the collaboration live.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional

from orchestra.agents.base import Agent, AgentResult
from orchestra.agents.roles import build_default_agents
from orchestra.llm import LLMBackend
from orchestra.observability import M, get_logger
from orchestra.rag.pipeline import Passage, RAGPipeline
from orchestra.reliability import (
    Budget,
    BudgetExceeded,
    estimate_cost_usd,
    estimate_tokens,
)

__all__ = ["Event", "OrchestratorResult", "Orchestrator"]

_log = get_logger("orchestra.orchestrator")


@dataclass
class Event:
    """A structured event emitted during a run, suitable for streaming to a UI."""

    type: str  # "start" | "agent_start" | "agent_message" | "round" | "final" | "error"
    role: str = ""
    content: str = ""
    round: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class OrchestratorResult:
    """The terminal result of an orchestrated run."""

    question: str
    answer: str
    passages: List[Passage]
    approved: bool
    rounds: int
    transcript: List[Event]

    def citations(self) -> List[Dict[str, object]]:
        """Return citation metadata for the passages used in the answer."""
        return [
            {
                "n": p.citation,
                "source": p.short_source,
                "score": round(p.score, 4),
                "text": p.text,
            }
            for p in self.passages
        ]


class Orchestrator:
    """Coordinates the agent roles, grounded by a shared RAG pipeline."""

    def __init__(
        self,
        llm: LLMBackend,
        rag: RAGPipeline,
        *,
        strategy: str = "linear",
        k: int = 4,
        max_rounds: int = 3,
        hybrid: bool = False,
        max_cost_usd: float = 1.0,
        agents: Optional[Dict[str, Agent]] = None,
    ) -> None:
        self.llm = llm
        self.rag = rag
        self.strategy = strategy
        self.k = k
        self.max_rounds = max(1, max_rounds)
        self.hybrid = hybrid
        self.max_cost_usd = max_cost_usd
        self.agents: Dict[str, Agent] = agents or build_default_agents(llm, rag, k=k, hybrid=hybrid)

    def _charge_llm(self, role: str, prompt: str, output: str, budget: Budget) -> None:
        """Estimate tokens/cost for one agent step, record metrics, and charge budget."""
        in_tokens = estimate_tokens(prompt)
        out_tokens = estimate_tokens(output)
        backend = getattr(self.llm, "name", "mock")
        cost = estimate_cost_usd(backend, in_tokens, out_tokens)
        M.llm_tokens.labels(role=role, kind="input").inc(in_tokens)
        M.llm_tokens.labels(role=role, kind="output").inc(out_tokens)
        M.llm_cost.labels(backend=backend).inc(cost)
        # Cost budget is advisory for the mock (cost 0); real backends enforce it.
        budget.charge_cost(cost)

    # -- public API -------------------------------------------------------
    def run(self, question: str) -> OrchestratorResult:
        """Run the full orchestration and return the terminal result."""
        result: Optional[OrchestratorResult] = None
        for event in self.stream(question):
            if event.type == "final":
                result = event.metadata.get("result")  # type: ignore[assignment]
        if result is None:  # pragma: no cover - defensive
            raise RuntimeError("Orchestration produced no final result")
        return result

    def stream(self, question: str) -> Iterator[Event]:
        """Run the orchestration, yielding structured events as they happen."""
        if self.strategy == "blackboard":
            yield from self._run_blackboard(question)
        else:
            yield from self._run_linear(question)

    # -- linear strategy --------------------------------------------------
    def _run_linear(self, question: str) -> Iterator[Event]:
        transcript: List[Event] = []

        def emit(event: Event) -> Event:
            transcript.append(event)
            return event

        yield emit(Event(type="start", content=question, metadata={"strategy": "linear"}))

        # 1. Plan
        yield emit(Event(type="agent_start", role="planner"))
        plan = self.agents["planner"].run(question)
        yield emit(Event(type="agent_message", role="planner", content=plan.content))

        # 2. Research (grounded retrieval)
        yield emit(Event(type="agent_start", role="researcher"))
        research: AgentResult = self.agents["researcher"].run(question)
        passages = research.passages
        yield emit(
            Event(
                type="agent_message",
                role="researcher",
                content=research.content,
                metadata={"num_passages": len(passages)},
            )
        )

        # 3. Synthesize -> 4. Critic -> revise loop (under a per-query budget)
        budget = Budget(max_rounds=self.max_rounds, max_cost_usd=self.max_cost_usd)
        evidence = research.content
        answer = ""
        approved = False
        rounds = 0
        budget_hit = False
        for rnd in range(1, self.max_rounds + 1):
            rounds = rnd
            try:
                budget.charge_round()
            except BudgetExceeded as exc:
                budget_hit = True
                yield emit(Event(type="budget", round=rnd, content=str(exc)))
                break
            yield emit(Event(type="round", round=rnd, metadata={"strategy": "linear"}))
            # Heartbeat so long-running streams keep the connection warm.
            yield emit(Event(type="heartbeat", round=rnd))

            yield emit(Event(type="agent_start", role="synthesizer", round=rnd))
            synth = self.agents["synthesizer"].run(
                question, passages=passages, evidence=evidence
            )
            answer = synth.content
            self._charge_llm("synthesizer", evidence + question, answer, budget)
            yield emit(
                Event(type="agent_message", role="synthesizer", content=answer, round=rnd)
            )

            yield emit(Event(type="agent_start", role="critic", round=rnd))
            critique = self.agents["critic"].run(
                question, draft=answer, passages=passages
            )
            self._charge_llm("critic", answer, critique.content, budget)
            approved = bool(critique.metadata.get("approved"))
            yield emit(
                Event(
                    type="agent_message",
                    role="critic",
                    content=critique.content,
                    round=rnd,
                    metadata={"approved": approved},
                )
            )
            if approved:
                break
            # Feed the critique back into the evidence for the next revision.
            evidence = f"{research.content}\n\nCritic feedback to address:\n{critique.content}"

        M.query_rounds.observe(rounds)
        _log.info("linear_run_complete", rounds=rounds, approved=approved, budget_hit=budget_hit)
        result = OrchestratorResult(
            question=question,
            answer=answer,
            passages=passages,
            approved=approved,
            rounds=rounds,
            transcript=list(transcript),
        )
        yield emit(
            Event(
                type="final",
                role="synthesizer",
                content=answer,
                round=rounds,
                metadata={"approved": approved, "result": result},
            )
        )

    # -- blackboard strategy ----------------------------------------------
    def _run_blackboard(self, question: str) -> Iterator[Event]:
        transcript: List[Event] = []
        blackboard: List[str] = []

        def emit(event: Event) -> Event:
            transcript.append(event)
            return event

        def board_text() -> str:
            return "\n\n".join(blackboard)

        yield emit(Event(type="start", content=question, metadata={"strategy": "blackboard"}))

        # Seed the board with a plan.
        yield emit(Event(type="agent_start", role="planner"))
        plan = self.agents["planner"].run(question)
        blackboard.append(f"[planner]\n{plan.content}")
        yield emit(Event(type="agent_message", role="planner", content=plan.content))

        # Research once; passages are stable grounding for the whole loop.
        yield emit(Event(type="agent_start", role="researcher"))
        research = self.agents["researcher"].run(question)
        passages = research.passages
        blackboard.append(f"[researcher]\n{research.content}")
        yield emit(
            Event(
                type="agent_message",
                role="researcher",
                content=research.content,
                metadata={"num_passages": len(passages)},
            )
        )

        answer = ""
        approved = False
        rounds = 0
        for rnd in range(1, self.max_rounds + 1):
            rounds = rnd
            yield emit(Event(type="round", round=rnd, metadata={"strategy": "blackboard"}))

            # Synthesizer reads the whole board as evidence.
            yield emit(Event(type="agent_start", role="synthesizer", round=rnd))
            synth = self.agents["synthesizer"].run(
                question, passages=passages, evidence=board_text()
            )
            answer = synth.content
            blackboard.append(f"[synthesizer round {rnd}]\n{answer}")
            yield emit(
                Event(type="agent_message", role="synthesizer", content=answer, round=rnd)
            )

            # Critic posts to the board.
            yield emit(Event(type="agent_start", role="critic", round=rnd))
            critique = self.agents["critic"].run(
                question, draft=answer, passages=passages
            )
            approved = bool(critique.metadata.get("approved"))
            blackboard.append(f"[critic round {rnd}]\n{critique.content}")
            yield emit(
                Event(
                    type="agent_message",
                    role="critic",
                    content=critique.content,
                    round=rnd,
                    metadata={"approved": approved},
                )
            )
            if approved:
                break

        result = OrchestratorResult(
            question=question,
            answer=answer,
            passages=passages,
            approved=approved,
            rounds=rounds,
            transcript=list(transcript),
        )
        yield emit(
            Event(
                type="final",
                role="synthesizer",
                content=answer,
                round=rounds,
                metadata={"approved": approved, "result": result},
            )
        )
