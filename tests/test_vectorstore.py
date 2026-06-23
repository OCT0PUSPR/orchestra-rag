"""Tests for the NumpyStore: ranking, persistence, upsert."""

from __future__ import annotations

import numpy as np

from orchestra.rag.embeddings import HashingEmbedder
from orchestra.rag.vectorstore import NumpyStore


def _store_with(texts):
    emb = HashingEmbedder(dimension=512)
    store = NumpyStore()
    vecs = emb.embed(texts)
    ids = [f"id-{i}" for i in range(len(texts))]
    metas = [{"source": f"src-{i}"} for i in range(len(texts))]
    store.add(ids, vecs, texts, metas)
    return emb, store


def test_similarity_ranking():
    texts = [
        "the warehouse robot battery lasts nine hours",
        "robots reserve cells to avoid collisions in aisles",
        "employees get twenty five vacation days each year",
    ]
    emb, store = _store_with(texts)
    query = emb.embed(["how long does the robot battery last"])[0]
    results = store.similarity_search(query, k=3)
    assert len(results) == 3
    # The battery sentence must rank first.
    assert "battery" in results[0].text
    # Scores must be in descending order.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_k_is_clamped_to_size():
    emb, store = _store_with(["a only document about robots"])
    query = emb.embed(["robots"])[0]
    results = store.similarity_search(query, k=10)
    assert len(results) == 1


def test_empty_store_returns_nothing():
    store = NumpyStore()
    results = store.similarity_search(np.zeros(512, dtype=np.float32), k=4)
    assert results == []


def test_upsert_replaces_not_duplicates():
    emb = HashingEmbedder(dimension=256)
    store = NumpyStore()
    store.add(["x"], emb.embed(["first"]), ["first"], [{"source": "a"}])
    store.add(["x"], emb.embed(["second version"]), ["second version"], [{"source": "a"}])
    assert len(store) == 1
    results = store.similarity_search(emb.embed(["second version"])[0], k=1)
    assert results[0].text == "second version"


def test_persistence_round_trip(tmp_path):
    emb = HashingEmbedder(dimension=256)
    texts = ["robot battery lasts nine hours", "vacation days policy"]
    store = NumpyStore(persist_dir=tmp_path)
    store.add(["a", "b"], emb.embed(texts), texts, [{"source": "1"}, {"source": "2"}])
    store.persist()

    reopened = NumpyStore(persist_dir=tmp_path)
    assert len(reopened) == 2
    results = reopened.similarity_search(emb.embed(["battery life"])[0], k=1)
    assert "battery" in results[0].text
