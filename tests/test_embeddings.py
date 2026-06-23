"""Tests for the deterministic hashing embedder."""

from __future__ import annotations

import numpy as np

from orchestra.rag.embeddings import HashingEmbedder, get_embedder


def test_dimension_and_shape():
    emb = HashingEmbedder(dimension=256)
    out = emb.embed(["hello world", "another document here"])
    assert out.shape == (2, 256)
    assert out.dtype == np.float32


def test_determinism():
    emb1 = HashingEmbedder(dimension=512)
    emb2 = HashingEmbedder(dimension=512)
    a = emb1.embed(["the quick brown fox jumps"])
    b = emb2.embed(["the quick brown fox jumps"])
    assert np.allclose(a, b)


def test_vectors_are_l2_normalized():
    emb = HashingEmbedder(dimension=256)
    out = emb.embed(["warehouse robot battery", "vacation policy parental leave"])
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_similar_texts_are_closer_than_dissimilar():
    emb = HashingEmbedder(dimension=512)
    vecs = emb.embed(
        [
            "the robot battery lasts nine hours and swaps quickly",
            "battery life and fast swapping of the robot pack",
            "employees receive twenty five days of paid vacation",
        ]
    )
    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])
    assert sim_related > sim_unrelated


def test_empty_input_returns_empty_matrix():
    emb = HashingEmbedder(dimension=128)
    out = emb.embed([])
    assert out.shape == (0, 128)


def test_get_embedder_hashing():
    emb = get_embedder("hashing", dimension=64)
    assert emb.dimension == 64
    assert emb.embed(["x"]).shape == (1, 64)
