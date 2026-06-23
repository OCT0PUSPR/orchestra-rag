"""Concrete agent roles: Planner, Researcher, Coder, Critic, Synthesizer.

Each role is a real prompt-driven specialist. The Researcher actually queries the
shared RAG pipeline and returns grounded passages; the Synthesizer produces the
final, citation-bearing answer; the Critic reviews it for grounding and
correctness.
"""

from __future__ import annotations

from typing import List, Optional

from orchestra.agents.base import Agent, AgentResult
from orchestra.llm import LLMBackend
from orchestra.rag.pipeline import Passage, RAGPipeline

__all__ = [
    "PlannerAgent",
    "ResearcherAgent",
    "CoderAgent",
    "CriticAgent",
    "SynthesizerAgent",
    "build_default_agents",
]


class PlannerAgent(Agent):
    """Decomposes a user query into ordered, answerable subtasks."""

    role = "planner"

    def default_system_prompt(self) -> str:
        return (
            "You are the Planner in a multi-agent system. Decompose the user's "
            "question into a short, ordered list of concrete subtasks that the "
            "downstream Researcher, Synthesizer, and Critic agents can execute. "
            "Keep it to at most five steps. Do not answer the question yourself."
        )

    def run(self, task: str, *, scratchpad: Optional[str] = None) -> AgentResult:
        prompt = f"QUESTION: {task}\n\nProduce the plan."
        content = self.complete(prompt)
        return AgentResult(role=self.role, content=content)


class ResearcherAgent(Agent):
    """Queries the shared RAG knowledge base and gathers cited evidence."""

    role = "researcher"

    def __init__(self, llm: LLMBackend, *, rag: Optional[RAGPipeline] = None, k: int = 4, **kw) -> None:
        super().__init__(llm, rag=rag, **kw)
        self.k = k

    def default_system_prompt(self) -> str:
        return (
            "You are the Researcher in a multi-agent system. You are given a "
            "question and a numbered CONTEXT block of passages retrieved from a "
            "knowledge base. Extract the facts that bear on the question and list "
            "them, citing the passage number in square brackets for each fact. "
            "Never invent facts that are not in the context."
        )

    def run(self, task: str, *, scratchpad: Optional[str] = None) -> AgentResult:
        passages: List[Passage] = self.retrieve(task, k=self.k)
        context = RAGPipeline.build_context(passages)
        prompt = f"QUESTION: {task}\n\nCONTEXT:\n{context}"
        content = self.complete(prompt)
        return AgentResult(role=self.role, content=content, passages=passages)


class CoderAgent(Agent):
    """Writes or edits code when the task calls for it, grounded in context."""

    role = "coder"

    def default_system_prompt(self) -> str:
        return (
            "You are the Coder in a multi-agent system. When the task requires "
            "writing or editing code, produce clear, correct, well-documented code "
            "in a fenced block. Ground any domain-specific behaviour in the "
            "provided CONTEXT and cite passage numbers in comments where relevant. "
            "If the task does not require code, say so briefly."
        )

    def run(self, task: str, *, scratchpad: Optional[str] = None) -> AgentResult:
        context = scratchpad or ""
        prompt = f"QUESTION: {task}\n\nCONTEXT:\n{context}"
        content = self.complete(prompt)
        return AgentResult(role=self.role, content=content)


class SynthesizerAgent(Agent):
    """Writes the final, citation-bearing answer from gathered evidence."""

    role = "synthesizer"

    def default_system_prompt(self) -> str:
        return (
            "You are the Synthesizer in a multi-agent system. Using only the "
            "numbered CONTEXT passages and the Researcher's EVIDENCE, write a "
            "concise, accurate answer to the question. Every factual claim must "
            "carry an inline citation like [1] pointing at the passage that "
            "supports it. If the context does not contain the answer, say so "
            "plainly rather than guessing."
        )

    def run(
        self,
        task: str,
        *,
        scratchpad: Optional[str] = None,
        passages: Optional[List[Passage]] = None,
        evidence: str = "",
    ) -> AgentResult:
        passages = passages or []
        context = RAGPipeline.build_context(passages)
        prompt = (
            f"QUESTION: {task}\n\n"
            f"EVIDENCE:\n{evidence}\n\n"
            f"CONTEXT:\n{context}"
        )
        content = self.complete(prompt)
        return AgentResult(role=self.role, content=content, passages=passages)


class CriticAgent(Agent):
    """Reviews a draft answer for correctness and grounding."""

    role = "critic"

    def default_system_prompt(self) -> str:
        return (
            "You are the Critic in a multi-agent system. Review the DRAFT answer "
            "against the numbered CONTEXT. Check that every claim is supported by a "
            "citation into the context and that nothing is fabricated. If the draft "
            "is correct and well-grounded, reply beginning with 'APPROVED'. "
            "Otherwise reply beginning with 'NEEDS REVISION' and list the specific "
            "problems to fix."
        )

    def run(
        self,
        task: str,
        *,
        scratchpad: Optional[str] = None,
        draft: str = "",
        passages: Optional[List[Passage]] = None,
    ) -> AgentResult:
        passages = passages or []
        context = RAGPipeline.build_context(passages)
        prompt = (
            f"QUESTION: {task}\n\n"
            f"DRAFT:\n{draft}\n\n"
            f"CONTEXT:\n{context}"
        )
        content = self.complete(prompt)
        approved = content.strip().upper().startswith("APPROVED")
        return AgentResult(
            role=self.role,
            content=content,
            passages=passages,
            metadata={"approved": approved},
        )


def build_default_agents(
    llm: LLMBackend,
    rag: RAGPipeline,
    *,
    k: int = 4,
) -> dict:
    """Construct one instance of every default role, wired to ``llm`` and ``rag``."""
    return {
        "planner": PlannerAgent(llm, rag=rag),
        "researcher": ResearcherAgent(llm, rag=rag, k=k),
        "coder": CoderAgent(llm, rag=rag),
        "synthesizer": SynthesizerAgent(llm, rag=rag),
        "critic": CriticAgent(llm, rag=rag),
    }
