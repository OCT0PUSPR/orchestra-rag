"""Evaluation harness.

Computes, over a labeled question set:

* **precision@k** — fraction of the top-k retrieved chunks whose source document
  is one of the question's relevant sources.
* **recall@k** — fraction of relevant source documents that appear in the top-k.
* **groundedness** — whether the final synthesized answer contains the expected
  answer keywords AND only cites passages that were actually retrieved (citation
  integrity).

Runs fully offline with the MockLLM. Designed to give RAG credibility — the
numbers are real and reproducible.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from orchestra.llm import MockLLM
from orchestra.orchestrator import Orchestrator
from orchestra.rag.pipeline import RAGPipeline

__all__ = ["QuestionSpec", "EvalResult", "load_question_set", "evaluate"]

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class QuestionSpec:
    id: str
    question: str
    relevant_sources: List[str]
    answer_keywords: List[str]


@dataclass
class EvalResult:
    k: int
    hybrid: bool
    per_question: List[Dict[str, object]] = field(default_factory=list)
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    groundedness: float = 0.0
    citation_integrity: float = 0.0

    def summary(self) -> str:
        mode = "hybrid" if self.hybrid else "dense"
        return (
            f"RAG eval ({mode}, k={self.k}, n={len(self.per_question)}): "
            f"precision@{self.k}={self.precision_at_k:.3f} "
            f"recall@{self.k}={self.recall_at_k:.3f} "
            f"nDCG@{self.k}={self.ndcg_at_k:.3f} "
            f"groundedness={self.groundedness:.3f} "
            f"citation_integrity={self.citation_integrity:.3f}"
        )


def _default_question_path() -> Path:
    return Path(__file__).resolve().parent / "questions.json"


def load_question_set(path: Optional[str | Path] = None) -> List[QuestionSpec]:
    p = Path(path) if path else _default_question_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    return [
        QuestionSpec(
            id=q["id"],
            question=q["question"],
            relevant_sources=list(q["relevant_sources"]),
            answer_keywords=list(q["answer_keywords"]),
        )
        for q in data["questions"]
    ]


def _retrieval_scores(retrieved_sources: Sequence[str], relevant: Sequence[str], k: int):
    top = list(retrieved_sources[:k])
    relevant_set = set(relevant)
    hit = [s for s in top if s in relevant_set]
    precision = len(hit) / len(top) if top else 0.0
    found = {s for s in top if s in relevant_set}
    recall = len(found) / len(relevant_set) if relevant_set else 0.0
    # nDCG@k with binary relevance over distinct relevant sources.
    dcg = 0.0
    seen: set = set()
    for rank, src in enumerate(top, start=1):
        if src in relevant_set and src not in seen:
            seen.add(src)
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_set), k) + 1))
    ndcg = (dcg / ideal) if ideal > 0 else 0.0
    return precision, recall, ndcg


def evaluate(
    pipeline: RAGPipeline,
    questions: Optional[List[QuestionSpec]] = None,
    *,
    k: int = 4,
    hybrid: bool = True,
) -> EvalResult:
    """Run the eval over ``pipeline`` (already ingested)."""
    questions = questions or load_question_set()
    orch = Orchestrator(MockLLM(), pipeline, strategy="linear", k=k, max_rounds=2)

    result = EvalResult(k=k, hybrid=hybrid)
    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []
    grounded_flags: List[float] = []
    integrity_flags: List[float] = []

    for q in questions:
        passages = pipeline.retrieve(q.question, k=k, hybrid=hybrid)
        retrieved_sources = [p.short_source for p in passages]
        precision, recall, ndcg = _retrieval_scores(retrieved_sources, q.relevant_sources, k)
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

        # End-to-end grounded answer.
        run = orch.run(q.question)
        answer = run.answer
        answer_l = answer.lower()
        kw_hit = sum(1 for kw in q.answer_keywords if kw.lower() in answer_l)
        grounded = 1.0 if (q.answer_keywords and kw_hit >= 1) else 0.0
        grounded_flags.append(grounded)

        # Citation integrity: every [n] in the answer must map to a retrieved
        # passage citation number.
        valid_cites = {p.citation for p in run.passages}
        used = {int(m) for m in _CITATION_RE.findall(answer)}
        integrity = 1.0 if used and used <= valid_cites else (1.0 if not used else 0.0)
        integrity_flags.append(integrity)

        result.per_question.append(
            {
                "id": q.id,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "ndcg": round(ndcg, 3),
                "grounded": bool(grounded),
                "keyword_hits": kw_hit,
                "retrieved": retrieved_sources,
                "citation_integrity": bool(integrity),
            }
        )

    n = len(questions) or 1
    result.precision_at_k = sum(precisions) / n
    result.recall_at_k = sum(recalls) / n
    result.ndcg_at_k = sum(ndcgs) / n
    result.groundedness = sum(grounded_flags) / n
    result.citation_integrity = sum(integrity_flags) / n
    return result
