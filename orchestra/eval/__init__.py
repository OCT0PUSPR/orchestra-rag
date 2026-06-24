"""RAG evaluation harness: retrieval precision@k / recall@k + groundedness."""

from __future__ import annotations

from orchestra.eval.harness import (
    EvalResult,
    QuestionSpec,
    evaluate,
    load_question_set,
)

__all__ = ["EvalResult", "QuestionSpec", "evaluate", "load_question_set"]
