"""An HNSW (Hierarchical Navigable Small World) index implemented from scratch.

Pure numpy + Python stdlib — no torch, no faiss, no hnswlib. This is a faithful
implementation of the Malkov & Yashunin algorithm:

* Each inserted node gets a random maximum level ``floor(-ln(U) * mL)``.
* Insertion greedily descends from the entry point through upper layers, then at
  each layer at/below the node level runs a beam search (``ef_construction``)
  and connects the node to its ``M`` nearest neighbours, with neighbour-degree
  pruning (``M`` on upper layers, ``2*M`` on layer 0).
* Search descends greedily through upper layers (ef=1) then beam-searches layer 0
  with ``ef`` to return the top-k.

Distance is cosine *distance* on L2-normalized vectors (so smaller = closer); we
normalize on add. ``query`` returns ``(ids, scores)`` where score is cosine
similarity in ``[-1, 1]`` to match the rest of the stack.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["HNSWIndex"]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


@dataclass
class HNSWIndex:
    """A cosine-similarity HNSW index over float32 vectors.

    Args:
        dim: Vector dimension.
        m: Max neighbours per node on layers > 0 (layer 0 gets ``2*m``).
        ef_construction: Beam width during insertion.
        ef_search: Default beam width during search.
        seed: RNG seed for reproducible level assignment.
    """

    dim: int
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 64
    seed: int = 42

    _vectors: List[np.ndarray] = field(default_factory=list)
    _labels: List[str] = field(default_factory=list)
    # graph[layer][node_idx] -> set of neighbour idxs
    _graph: List[Dict[int, set]] = field(default_factory=list)
    _entry: Optional[int] = None
    _max_level: int = 0
    _rng: random.Random = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._ml = 1.0 / math.log(2.0)  # level multiplier

    # -- internals ---------------------------------------------------------
    def _random_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random() + 1e-12) * self._ml))

    def _distance(self, a: int, vec: np.ndarray) -> float:
        # cosine distance on normalized vectors = 1 - dot.
        return 1.0 - float(np.dot(self._vectors[a], vec))

    def _neighbors(self, layer: int, node: int) -> set:
        return self._graph[layer].setdefault(node, set())

    def _search_layer(
        self, query: np.ndarray, entry_points: List[int], ef: int, layer: int
    ) -> List[Tuple[float, int]]:
        """Beam search one layer. Returns a list of ``(distance, node)`` (a heap)."""
        visited = set(entry_points)
        # candidates: min-heap by distance; results: max-heap (store neg dist).
        candidates: List[Tuple[float, int]] = []
        results: List[Tuple[float, int]] = []
        for ep in entry_points:
            d = self._distance(ep, query)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))
        while candidates:
            dist, node = heapq.heappop(candidates)
            worst = -results[0][0]
            if dist > worst and len(results) >= ef:
                break
            for neigh in self._neighbors(layer, node):
                if neigh in visited:
                    continue
                visited.add(neigh)
                d = self._distance(neigh, query)
                worst = -results[0][0]
                if d < worst or len(results) < ef:
                    heapq.heappush(candidates, (d, neigh))
                    heapq.heappush(results, (-d, neigh))
                    if len(results) > ef:
                        heapq.heappop(results)
        return [(-nd, n) for nd, n in results]

    def _select_neighbors(
        self, candidates: List[Tuple[float, int]], m: int
    ) -> List[int]:
        """Simple heuristic: take the ``m`` closest candidates."""
        candidates = sorted(candidates, key=lambda x: x[0])
        return [n for _, n in candidates[:m]]

    # -- public API --------------------------------------------------------
    def add(self, label: str, vector: np.ndarray) -> None:
        """Insert one labelled vector."""
        vec = _normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        node = len(self._vectors)
        self._vectors.append(vec)
        self._labels.append(label)
        level = self._random_level()

        # Grow the per-layer graph structures as needed.
        while len(self._graph) <= level:
            self._graph.append({})

        if self._entry is None:
            self._entry = node
            self._max_level = level
            for lyr in range(level + 1):
                self._neighbors(lyr, node)
            return

        ep = [self._entry]
        # Descend from the top down to level+1 with ef=1 (greedy).
        for lyr in range(self._max_level, level, -1):
            res = self._search_layer(vec, ep, ef=1, layer=lyr)
            ep = [min(res, key=lambda x: x[0])[1]] if res else ep

        # Connect on layers level..0.
        for lyr in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(vec, ep, ef=self.ef_construction, layer=lyr)
            m_lyr = self.m * 2 if lyr == 0 else self.m
            neighbors = self._select_neighbors(found, m_lyr)
            for n in neighbors:
                self._neighbors(lyr, node).add(n)
                self._neighbors(lyr, n).add(node)
                # Prune the neighbour's degree if over budget.
                if len(self._neighbors(lyr, n)) > m_lyr:
                    cand = [(self._distance(x, self._vectors[n]), x) for x in self._neighbors(lyr, n)]
                    keep = set(self._select_neighbors(cand, m_lyr))
                    self._graph[lyr][n] = keep
            ep = [n for _, n in found] or ep

        if level > self._max_level:
            self._max_level = level
            self._entry = node

    def query(self, vector: np.ndarray, k: int = 10, *, ef: Optional[int] = None) -> List[Tuple[str, float]]:
        """Return the ``k`` nearest labels with cosine similarity scores."""
        if self._entry is None or not self._vectors:
            return []
        vec = _normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        ef = ef or max(self.ef_search, k)
        ep = [self._entry]
        for lyr in range(self._max_level, 0, -1):
            res = self._search_layer(vec, ep, ef=1, layer=lyr)
            ep = [min(res, key=lambda x: x[0])[1]] if res else ep
        found = self._search_layer(vec, ep, ef=ef, layer=0)
        found.sort(key=lambda x: x[0])
        out: List[Tuple[str, float]] = []
        for dist, node in found[:k]:
            out.append((self._labels[node], 1.0 - dist))  # distance -> similarity
        return out

    def __len__(self) -> int:
        return len(self._vectors)
