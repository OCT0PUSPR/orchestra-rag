"""LLM backends behind a single :class:`LLMBackend` protocol.

Backends are interchangeable:

* :class:`MockLLM` — fully offline, deterministic, no API key, used by tests and
  the default demo. It is *not* a dumb echo: it inspects the role-specific system
  prompt and the retrieved context to produce role-appropriate, grounded output
  (including real inline citations for the synthesizer).
* :class:`AnthropicLLM` — the primary production backend, using the official
  Anthropic SDK with adaptive thinking.
* :class:`HuggingFaceLLM` — a secondary backend hitting the HF Inference API over
  httpx with an open instruct model.

The Anthropic and HF backends import their dependencies lazily so that importing
this module (and running the offline test suite) never requires them.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Protocol, runtime_checkable

__all__ = [
    "Message",
    "LLMBackend",
    "MockLLM",
    "AnthropicLLM",
    "HuggingFaceLLM",
    "get_llm",
]


class Message:
    """A single chat message."""

    __slots__ = ("role", "content")

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@runtime_checkable
class LLMBackend(Protocol):
    """The single interface every backend implements."""

    name: str

    def complete(
        self,
        system: str,
        messages: List[Message],
        *,
        max_tokens: int = 1024,
    ) -> str:
        """Return the assistant's text completion."""
        ...


# --------------------------------------------------------------------------- #
# MockLLM — offline, deterministic, role-aware                                 #
# --------------------------------------------------------------------------- #

_ROLE_MARKER_RE = re.compile(r"ROLE:\s*([A-Za-z_]+)")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]")


def _detect_role(system: str) -> str:
    match = _ROLE_MARKER_RE.search(system or "")
    if match:
        return match.group(1).lower()
    return "assistant"


def _last_user_text(messages: List[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""


def _extract_context_block(text: str) -> str:
    """Pull the citation-numbered context block out of a user message, if any."""
    marker = "CONTEXT:"
    idx = text.find(marker)
    if idx == -1:
        return ""
    return text[idx + len(marker) :].strip()


def _extract_question(text: str) -> str:
    marker = "QUESTION:"
    idx = text.find(marker)
    if idx == -1:
        # Fall back to the first non-empty line.
        for line in text.splitlines():
            if line.strip():
                return line.strip()
        return text.strip()
    rest = text[idx + len(marker) :]
    # Stop at the next labelled section if present.
    for stop in ("CONTEXT:", "DRAFT:", "EVIDENCE:", "SUBTASKS:", "Produce the plan."):
        cut = rest.find(stop)
        if cut != -1:
            rest = rest[:cut]
    # The question is the first paragraph; drop trailing instruction paragraphs.
    rest = rest.strip().split("\n\n")[0]
    return rest.strip()


def _parse_context_passages(context: str) -> List[tuple[int, str, str]]:
    """Return [(citation_number, source, text), ...] from a rendered context block."""
    passages: List[tuple[int, str, str]] = []
    # Split on the leading [n] markers.
    parts = re.split(r"\n(?=\[\d+\])", context)
    for part in parts:
        m = re.match(r"\[(\d+)\]\s*\(source:\s*([^)]*)\)\s*(.*)", part.strip(), re.DOTALL)
        if m:
            num = int(m.group(1))
            source = m.group(2).strip()
            body = " ".join(m.group(3).split())
            passages.append((num, source, body))
    return passages


def _keyword_score(question: str, text: str) -> int:
    q_words = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2}
    t_words = re.findall(r"[a-z0-9]+", text.lower())
    return sum(1 for w in t_words if w in q_words)


def _clean_sentence(sentence: str) -> str:
    """Strip leading markdown header markers (``#``, ``##``, ``**``) and bullets.

    Word-based chunking can fuse a markdown header onto the first sentence (no
    trailing period); this removes the heading noise so cited sentences read as
    real prose.
    """
    s = sentence.strip()
    # Drop a leading run of markdown header / bullet markers and their heading
    # words up to the first lowercase-started clause when a heading is detected.
    s = re.sub(r"^[#>*\-\s]+", "", s)
    return s.strip()


def _best_sentences(question: str, text: str, limit: int = 2) -> List[str]:
    raw = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    sentences = [_clean_sentence(s) for s in raw]
    sentences = [s for s in sentences if s]
    if not sentences:
        cleaned = _clean_sentence(text)
        return [cleaned] if cleaned else []

    def score(s: str) -> tuple[int, int]:
        # Primary: keyword overlap. Secondary: prefer fuller sentences over
        # short heading-like fragments, so cited text is informative.
        return (_keyword_score(question, s), min(len(s.split()), 40))

    scored = sorted(sentences, key=score, reverse=True)
    chosen = [s for s in scored[:limit] if _keyword_score(question, s) > 0]
    if not chosen:
        chosen = sentences[:1]
    return chosen


class MockLLM:
    """A deterministic, offline LLM that produces role-appropriate output.

    The orchestrator passes a ``ROLE: <name>`` marker in each system prompt; this
    backend keys off that marker to behave differently for the Planner,
    Researcher, Coder, Critic, and Synthesizer roles. The Synthesizer produces a
    genuinely grounded answer with real ``[n]`` citations pulled from the
    retrieved context, which is what makes the offline demo meaningful.
    """

    name = "mock"

    def complete(
        self,
        system: str,
        messages: List[Message],
        *,
        max_tokens: int = 1024,
    ) -> str:
        role = _detect_role(system)
        user_text = _last_user_text(messages)
        question = _extract_question(user_text)
        context = _extract_context_block(user_text)
        passages = _parse_context_passages(context)

        if role == "planner":
            return self._plan(question)
        if role == "researcher":
            return self._research(question, passages)
        if role == "coder":
            return self._code(question, passages)
        if role == "critic":
            return self._critique(question, user_text, passages)
        if role == "synthesizer":
            return self._synthesize(question, passages)
        return self._synthesize(question, passages)

    # -- per-role behaviours ----------------------------------------------
    def _plan(self, question: str) -> str:
        topic = question.rstrip("?.! ") or "the request"
        return (
            "Plan to answer the question by decomposing it into grounded steps:\n"
            f"1. Identify the key entities and facts referenced in: {topic}.\n"
            "2. Retrieve the most relevant passages from the knowledge base.\n"
            "3. Extract the specific facts that answer the question, with citations.\n"
            "4. Synthesize a concise, fully-cited answer.\n"
            "5. Review the answer for correctness and grounding."
        )

    def _research(self, question: str, passages: List[tuple[int, str, str]]) -> str:
        if not passages:
            return (
                "No grounded evidence was found in the knowledge base for this "
                "question. The answer cannot be supported by the provided sources."
            )
        ranked = sorted(passages, key=lambda p: _keyword_score(question, p[2]), reverse=True)
        lines: List[str] = ["Grounded evidence gathered from the knowledge base:"]
        for num, source, text in ranked:
            facts = _best_sentences(question, text, limit=1)
            fact = facts[0] if facts else text[:160]
            lines.append(f"- [{num}] (source: {source}) {fact}")
        return "\n".join(lines)

    def _code(self, question: str, passages: List[tuple[int, str, str]]) -> str:
        hint = passages[0][2][:80] if passages else "the requested behaviour"
        return (
            "```python\n"
            "def answer_query(query: str) -> str:\n"
            '    """Generated helper grounded in the retrieved knowledge base.\n'
            f"    Context hint: {hint!r}\n"
            '    """\n'
            "    # Replace with real logic; this is a deterministic mock scaffold.\n"
            "    return f\"Handling: {query}\"\n"
            "```\n"
            "Note: this is mock-generated scaffolding. With a real LLM backend the "
            "Coder role writes task-specific code grounded in the retrieved context."
        )

    def _critique(
        self,
        question: str,
        full_user_text: str,
        passages: List[tuple[int, str, str]],
    ) -> str:
        has_citation = bool(_CITATION_RE.search(full_user_text))
        grounded = bool(passages)
        if grounded and has_citation:
            return (
                "APPROVED. The draft answers the question and every claim is "
                "supported by a citation into the retrieved context. No revisions "
                "required."
            )
        issues: List[str] = []
        if not grounded:
            issues.append("the draft is not grounded in any retrieved passage")
        if not has_citation:
            issues.append("the draft lacks inline citations")
        return (
            "NEEDS REVISION. Problems detected: "
            + "; ".join(issues)
            + ". Revise to add grounded, cited claims."
        )

    def _synthesize(self, question: str, passages: List[tuple[int, str, str]]) -> str:
        if not passages:
            return (
                "I could not find information in the knowledge base to answer this "
                "question, so I cannot provide a grounded answer."
            )
        # Score every candidate sentence across all passages and take the most
        # question-relevant ones globally, keeping each sentence's citation.
        candidates: List[tuple[int, str, int]] = []  # (score, sentence, citation)
        for num, _source, text in passages:
            for sentence in _best_sentences(question, text, limit=3):
                candidates.append((_keyword_score(question, sentence), sentence, num))
        candidates.sort(key=lambda c: c[0], reverse=True)

        sentences: List[str] = []
        used: set[int] = set()
        seen_text: set[str] = set()
        for score, sentence, num in candidates:
            if score <= 0:
                continue
            key = sentence.lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            clean = sentence.strip().rstrip(".")
            sentences.append(f"{clean} [{num}].")
            used.add(num)
            if len(sentences) >= 3:
                break

        if not sentences:
            num, _source, text = passages[0]
            sentences.append(f"{text[:200].strip()} [{num}].")
            used.add(num)
        answer = " ".join(sentences)
        cite_list = ", ".join(f"[{n}]" for n in sorted(used))
        return f"{answer}\n\nSources: {cite_list}"


# --------------------------------------------------------------------------- #
# AnthropicLLM — primary production backend                                    #
# --------------------------------------------------------------------------- #


class AnthropicLLM:
    """Anthropic Claude backend using the official SDK with adaptive thinking."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        api_key: Optional[str] = None,
        use_thinking: bool = True,
    ) -> None:
        self.model = model
        self.use_thinking = use_thinking
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AnthropicLLM requires the `anthropic` package. Install it with "
                "`pip install anthropic`, or use the mock backend (--backend mock)."
            ) from exc
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or use --backend mock to "
                "run fully offline."
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        system: str,
        messages: List[Message],
        *,
        max_tokens: int = 1024,
    ) -> str:
        client = self._ensure_client()
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [m.as_dict() for m in messages],
        }
        if self.use_thinking:
            # Adaptive thinking is the recommended mode for Claude 4.6+ models.
            kwargs["thinking"] = {"type": "adaptive"}
        response = client.messages.create(**kwargs)
        parts: List[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts).strip()


# --------------------------------------------------------------------------- #
# HuggingFaceLLM — secondary backend via the Inference API                     #
# --------------------------------------------------------------------------- #


class HuggingFaceLLM:
    """Hugging Face Inference API backend (open instruct models) via httpx."""

    name = "huggingface"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        api_token: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self._token = api_token or os.environ.get("HF_TOKEN")

    def _build_prompt(self, system: str, messages: List[Message]) -> List[dict]:
        chat = [{"role": "system", "content": system}]
        for m in messages:
            chat.append(m.as_dict())
        return chat

    def complete(
        self,
        system: str,
        messages: List[Message],
        *,
        max_tokens: int = 1024,
    ) -> str:
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "HuggingFaceLLM requires `httpx`. Install it with `pip install httpx`."
            ) from exc
        if not self._token:
            raise RuntimeError(
                "HF_TOKEN is not set. Export it, or use --backend mock to run offline."
            )
        url = "https://api-inference.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self._token}"}
        payload = {
            "model": self.model,
            "messages": self._build_prompt(system, messages),
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #


def get_llm(backend: str = "mock", **kwargs) -> LLMBackend:
    """Construct an LLM backend by name.

    Args:
        backend: ``"mock"`` (default), ``"anthropic"``, or ``"huggingface"``.
        **kwargs: Backend-specific constructor arguments (e.g. ``model``).
    """
    backend = (backend or "mock").lower()
    if backend == "mock":
        return MockLLM()
    if backend in {"anthropic", "claude"}:
        return AnthropicLLM(**kwargs)
    if backend in {"huggingface", "hf"}:
        return HuggingFaceLLM(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend!r}")
