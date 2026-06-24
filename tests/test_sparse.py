"""Tests for the pure-Python BM25 sparse index."""

from __future__ import annotations

from orchestra.rag.sparse import BM25Index


def _index():
    idx = BM25Index()
    idx.add_many(
        [
            ("d1", "the warehouse robot battery lasts nine hours and swaps fast"),
            ("d2", "robots reserve cells to avoid collisions in narrow aisles"),
            ("d3", "employees receive twenty five days of paid vacation each year"),
        ]
    )
    return idx


def test_bm25_ranks_relevant_doc_first():
    idx = _index()
    hits = idx.search("how long does the battery last", k=3)
    assert hits
    assert hits[0].doc_id == "d1"


def test_bm25_scores_descending():
    idx = _index()
    hits = idx.search("robots cells collisions", k=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_bm25_empty_query_and_index():
    assert BM25Index().search("anything", k=3) == []
    idx = _index()
    assert idx.search("", k=3) == []


def test_bm25_no_match_returns_empty():
    idx = _index()
    assert idx.search("quantum bananas xyzzy", k=3) == []


def test_bm25_k_clamped():
    idx = _index()
    hits = idx.search("vacation", k=99)
    assert len(hits) >= 1
    assert all(h.doc_id for h in hits)
