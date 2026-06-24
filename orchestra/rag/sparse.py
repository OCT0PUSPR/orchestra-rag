"""Pure-Python BM25 sparse retriever (zero heavy dependencies).

Implements Okapi BM25 over an in-memory inverted index. Used as the sparse arm
of hybrid search. If ``rank_bm25`` is installed it is *not* required — this
implementation is self-contained, deterministic, and unit-tested, so hybrid
retrieval works in the zero-dependency configuration.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

__all__ = ["BM25Index", "BM25Hit"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Hit:
    """A sparse-retrieval hit."""

    doc_id: str
    score: float


@dataclass
class BM25Index:
    """In-memory Okapi BM25 index.

    Args:
        k1: Term-frequency saturation parameter.
        b: Length-normalization parameter.
    """

    k1: float = 1.5
    b: float = 0.75
    _doc_ids: List[str] = field(default_factory=list)
    _doc_len: List[int] = field(default_factory=list)
    _tf: List[Dict[str, int]] = field(default_factory=list)
    _df: Dict[str, int] = field(default_factory=dict)
    _avgdl: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        """Add a single document to the index."""
        tokens = _tokenize(text)
        tf: Dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        self._doc_ids.append(doc_id)
        self._doc_len.append(len(tokens))
        self._tf.append(tf)
        for term in tf:
            self._df[term] = self._df.get(term, 0) + 1
        self._recompute_avgdl()

    def add_many(self, items: List[Tuple[str, str]]) -> None:
        """Add many ``(doc_id, text)`` pairs."""
        for doc_id, text in items:
            self.add(doc_id, text)

    def _recompute_avgdl(self) -> None:
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._doc_ids)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        # BM25+ style IDF, always non-negative.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> List[BM25Hit]:
        """Return the top-``k`` documents by BM25 score."""
        if not self._doc_ids:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scores: List[float] = [0.0] * len(self._doc_ids)
        for term in set(q_terms):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for i, tf in enumerate(self._tf):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                dl = self._doc_len[i]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0))
                scores[i] += idf * (f * (self.k1 + 1.0)) / denom if denom else 0.0
        ranked = sorted(
            range(len(self._doc_ids)),
            key=lambda i: scores[i],
            reverse=True,
        )
        k = max(1, min(k, len(self._doc_ids)))
        results: List[BM25Hit] = []
        for i in ranked[:k]:
            if scores[i] <= 0.0:
                continue
            results.append(BM25Hit(doc_id=self._doc_ids[i], score=float(scores[i])))
        return results

    def __len__(self) -> int:
        return len(self._doc_ids)
