"""The base :class:`Agent`: a role, a system prompt, an LLM, and optional RAG.

Every concrete role subclasses :class:`Agent` and supplies its own system prompt
and ``run`` logic. An agent may be granted access to a :class:`RAGPipeline`, in
which case it can retrieve grounded passages as a "tool".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from orchestra.llm import LLMBackend, Message
from orchestra.rag.pipeline import Passage, RAGPipeline


@dataclass
class AgentResult:
    """The output of an agent turn."""

    role: str
    content: str
    passages: List[Passage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Agent:
    """A prompt-driven specialist.

    Args:
        role: Stable role identifier (also injected as a ``ROLE:`` marker so the
            MockLLM and prompts can branch on it).
        system_prompt: The role's standing instructions.
        llm: The backend used for completions.
        rag: Optional shared RAG pipeline this agent may query.
        max_tokens: Default completion budget.
    """

    role: str = "agent"

    def __init__(
        self,
        llm: LLMBackend,
        *,
        rag: Optional[RAGPipeline] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        self.llm = llm
        self.rag = rag
        self.max_tokens = max_tokens
        self._system_prompt = system_prompt or self.default_system_prompt()

    # -- prompt -----------------------------------------------------------
    def default_system_prompt(self) -> str:
        """Override in subclasses with the role's standing instructions."""
        return "You are a helpful assistant."

    @property
    def system_prompt(self) -> str:
        """The system prompt with the machine-readable role marker prepended."""
        return f"ROLE: {self.role}\n\n{self._system_prompt}"

    # -- tools ------------------------------------------------------------
    def retrieve(self, query: str, k: int = 4) -> List[Passage]:
        """Tool: query the shared knowledge base for grounded passages."""
        if self.rag is None:
            return []
        return self.rag.retrieve(query, k=k)

    # -- completion -------------------------------------------------------
    def complete(self, user_content: str) -> str:
        """Run a single completion with this agent's system prompt."""
        return self.llm.complete(
            self.system_prompt,
            [Message("user", user_content)],
            max_tokens=self.max_tokens,
        )

    # -- main entry point -------------------------------------------------
    def run(self, task: str, *, scratchpad: Optional[str] = None) -> AgentResult:
        """Execute the agent on a task. Subclasses override this."""
        content = self.complete(task)
        return AgentResult(role=self.role, content=content)
