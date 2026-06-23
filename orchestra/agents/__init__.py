"""Specialized prompt-driven agent roles coordinated by the orchestrator."""

from __future__ import annotations

from orchestra.agents.base import Agent, AgentResult
from orchestra.agents.roles import (
    CoderAgent,
    CriticAgent,
    PlannerAgent,
    ResearcherAgent,
    SynthesizerAgent,
    build_default_agents,
)

__all__ = [
    "Agent",
    "AgentResult",
    "PlannerAgent",
    "ResearcherAgent",
    "CoderAgent",
    "CriticAgent",
    "SynthesizerAgent",
    "build_default_agents",
]
