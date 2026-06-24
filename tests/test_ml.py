"""Tests for the from-scratch ML package.

The torch-free parts (HNSW index, eval metrics, data synthesis, fallback
wiring) are always tested. The torch-dependent parts (transformer, bi-/cross-
encoder forward passes, training, ONNX) are tested only when torch is installed,
via ``pytest.importorskip`` — so this file imports and runs on the lightweight
(no-torch) CI path without failing.
"""

from __future__ import annotations

import numpy as np
import pytest

from orchestra.ml import HAS_TORCH, has_torch
from orchestra.ml.eval import ndcg_at_k, recall_at_k


# ---------------------------------------------------------------------------
# Torch-free: eval metrics
# ---------------------------------------------------------------------------
def test_recall_at_k_hit_and_miss():
    assert recall_at_k([3, 1, 2], gold_id=2, k=3) == 1.0
    assert recall_at_k([3, 1, 2], gold_id=2, k=2) == 0.0
    assert recall_at_k([], gold_id=2, k=5) == 0.0


def test_ndcg_at_k_position_weighting():
    # Gold at rank 1 -> 1/log2(2) = 1.0
    assert ndcg_at_k([2, 0, 1], gold_id=2, k=3) == pytest.approx(1.0)
    # Gold at rank 2 -> 1/log2(3)
    assert ndcg_at_k([0, 2, 1], gold_id=2, k=3) == pytest.approx(1.0 / np.log2(3))
    # Gold outside k -> 0
    assert ndcg_at_k([0, 1, 2], gold_id=2, k=2) == 0.0


def test_evaluate_ranker_perfect_and_random():
    from orchestra.ml.eval import evaluate_ranker

    queries = ["a", "b", "c"]
    gold = [0, 1, 2]
    # A perfect ranker always puts gold first.
    perfect = evaluate_ranker(lambda q, k: [gold[queries.index(q)]], queries, gold, k=1)
    assert perfect.recall == 1.0 and perfect.ndcg == 1.0
    # A ranker that never returns gold.
    miss = evaluate_ranker(lambda q, k: [99], queries, gold, k=1)
    assert miss.recall == 0.0 and miss.ndcg == 0.0


# ---------------------------------------------------------------------------
# Torch-free: HNSW index (pure numpy)
# ---------------------------------------------------------------------------
def test_hnsw_recall_matches_bruteforce():
    from orchestra.ml.hnsw import HNSWIndex

    rng = np.random.default_rng(0)
    n, d = 400, 32
    x = rng.standard_normal((n, d)).astype(np.float32)
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)

    idx = HNSWIndex(dim=d, m=16, ef_construction=200, ef_search=100, seed=0)
    for i in range(n):
        idx.add(f"d{i}", x[i])
    assert len(idx) == n

    # Brute-force top-5 for a handful of queries; HNSW should match most of them.
    q = rng.standard_normal((10, d)).astype(np.float32)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    sims = qn @ xn.T
    bf = np.argsort(-sims, axis=1)[:, :5]
    hits = 0
    for qi in range(10):
        got = {int(lbl[1:]) for lbl, _ in idx.query(q[qi], k=5)}
        hits += len(got & set(bf[qi].tolist()))
    # Expect very high overlap (graph is well-connected at this scale).
    assert hits / (10 * 5) >= 0.9


def test_hnsw_empty_query_returns_empty():
    from orchestra.ml.hnsw import HNSWIndex

    idx = HNSWIndex(dim=8)
    assert idx.query(np.zeros(8, dtype=np.float32), k=3) == []
    assert len(idx) == 0


def test_hnsw_self_retrieval_is_top1():
    from orchestra.ml.hnsw import HNSWIndex

    rng = np.random.default_rng(1)
    idx = HNSWIndex(dim=16, seed=0)
    vecs = rng.standard_normal((50, 16)).astype(np.float32)
    for i, v in enumerate(vecs):
        idx.add(f"v{i}", v)
    top_label, top_score = idx.query(vecs[7], k=1)[0]
    assert top_label == "v7"
    assert top_score == pytest.approx(1.0, abs=1e-4)  # cosine sim with itself


# ---------------------------------------------------------------------------
# Torch-free: synthetic data generation
# ---------------------------------------------------------------------------
def test_build_synthetic_pairs_from_corpus():
    from orchestra.app import default_corpus_dir
    from orchestra.ml.data import build_synthetic_pairs

    pairs, passages = build_synthetic_pairs(default_corpus_dir(), seed=0)
    assert len(pairs) > 20
    assert len(passages) > 5
    # Every pair's passage should be in the passage pool.
    pool = set(passages)
    assert all(p.passage in pool for p in pairs)
    # Queries are non-empty.
    assert all(p.query.strip() for p in pairs)


def test_train_val_split_is_deterministic_and_disjoint():
    from orchestra.ml.data import Pair, train_val_split

    pairs = [Pair(query=f"q{i}", passage=f"p{i}") for i in range(20)]
    a_tr, a_val = train_val_split(pairs, val_frac=0.25, seed=3)
    b_tr, b_val = train_val_split(pairs, val_frac=0.25, seed=3)
    assert [p.query for p in a_val] == [p.query for p in b_val]  # deterministic
    val_q = {p.query for p in a_val}
    tr_q = {p.query for p in a_tr}
    assert val_q.isdisjoint(tr_q)  # disjoint
    assert len(a_val) == 5


# ---------------------------------------------------------------------------
# Torch-free: availability + fallback wiring
# ---------------------------------------------------------------------------
def test_has_torch_flag_consistent():
    assert HAS_TORCH == has_torch()


def test_checkpoints_present_is_boolean():
    from orchestra.ml.adapters import checkpoints_present, default_checkpoint_dir

    assert isinstance(checkpoints_present(), bool)
    assert default_checkpoint_dir().name == "checkpoints"


def test_get_embedder_hashing_still_default_without_ml(monkeypatch):
    # Force the ML path to look unavailable; auto must fall back gracefully.
    import orchestra.rag.embeddings as emb

    monkeypatch.setattr(emb, "_ml_embedder", lambda: None)
    # Also make sentence-transformers look absent.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "sentence_transformers":
            raise ImportError("no st")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    e = emb.get_embedder("auto")
    assert type(e).__name__ == "HashingEmbedder"


def test_hnsw_store_roundtrip_via_vectorstore_factory():
    from orchestra.rag.vectorstore import get_vector_store

    store = get_vector_store("hnsw")
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((10, 16)).astype(np.float32)
    ids = [f"c{i}" for i in range(10)]
    texts = [f"text {i}" for i in range(10)]
    metas = [{"source": "x", "index": i} for i in range(10)]
    store.add(ids, vecs, texts, metas)
    assert len(store) == 10
    res = store.similarity_search(vecs[3], k=3)
    assert res and res[0].id == "c3"
    assert res[0].text == "text 3"


# ---------------------------------------------------------------------------
# Torch-dependent: only run when torch is installed
# ---------------------------------------------------------------------------
def test_transformer_and_biencoder_forward():
    pytest.importorskip("torch")
    import torch

    from orchestra.ml.bi_encoder import BiEncoder, info_nce_loss
    from orchestra.ml.transformer import EncoderConfig, mean_pool

    cfg = EncoderConfig(vocab_size=200, max_len=32, dim=32, depth=2, heads=2)
    model = BiEncoder(cfg)
    ids = torch.randint(1, 200, (4, 10))
    mask = torch.ones(4, 10, dtype=torch.long)
    emb = model(ids, mask)
    assert emb.shape == (4, 32)
    # Embeddings are unit-norm.
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-4)
    # InfoNCE loss is finite and positive.
    loss = info_nce_loss(emb, emb.roll(1, 0))
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    # mean_pool respects the mask (padded positions ignored).
    hidden = torch.randn(2, 5, 8)
    m = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    pooled = mean_pool(hidden, m)
    assert pooled.shape == (2, 8)
    assert torch.allclose(pooled[0], hidden[0, :2].mean(0), atol=1e-5)


def test_cross_encoder_forward_and_rank():
    pytest.importorskip("torch")
    import torch

    from orchestra.ml.cross_encoder import CrossEncoder
    from orchestra.ml.transformer import EncoderConfig

    cfg = EncoderConfig(vocab_size=200, max_len=32, dim=32, depth=2, heads=2)
    model = CrossEncoder(cfg)
    ids = torch.randint(1, 200, (3, 12))
    mask = torch.ones(3, 12, dtype=torch.long)
    seg = torch.zeros(3, 12, dtype=torch.long)
    logits = model(ids, mask, seg)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_bpe_tokenizer_train_encode_pair():
    pytest.importorskip("tokenizers")
    from orchestra.ml.tokenizer import CLS_ID, SEP_ID, BPETokenizer

    texts = ["the quick brown fox", "warehouse robots coordinate fleets", "battery swap takes seconds"]
    tok = BPETokenizer.train(texts * 5, vocab_size=200, min_frequency=1)
    ids, mask = tok.encode_batch(["quick fox", "robots"], max_len=16)
    assert len(ids) == 2 and len(ids[0]) == len(mask[0])
    assert ids[0][0] == CLS_ID  # [CLS] prefix
    pair_ids, pair_mask, pair_seg = tok.encode_pair("robots", "warehouse robots coordinate", max_len=32)
    assert pair_ids[0] == CLS_ID
    assert SEP_ID in pair_ids
    assert set(pair_seg) <= {0, 1}


def test_biencoder_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    import torch

    from orchestra.ml.bi_encoder import BiEncoder
    from orchestra.ml.transformer import EncoderConfig

    cfg = EncoderConfig(vocab_size=100, max_len=16, dim=16, depth=1, heads=2)
    model = BiEncoder(cfg)
    model.eval()  # disable dropout for a deterministic comparison
    path = tmp_path / "bi.pt"
    model.save(path)
    loaded = BiEncoder.load(path)  # load() sets eval() too
    ids = torch.randint(1, 100, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.long)
    with torch.no_grad():
        a = model(ids, mask)
        b = loaded(ids, mask)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.fixture(scope="module")
def tiny_tokenizer():
    pytest.importorskip("tokenizers")
    from orchestra.ml.tokenizer import BPETokenizer

    corpus = [
        "the atlas robot swaps its battery in under thirty seconds",
        "warehouse fleets are coordinated by the conductor platform",
        "approved production languages are rust python and typescript",
        "employees receive eighteen weeks of parental leave",
        "the picking arm handles items up to five kilograms",
        "nimbus robotics is headquartered in portland oregon",
    ]
    return BPETokenizer.train(corpus * 8, vocab_size=300, min_frequency=1)


def _tiny_pairs():
    from orchestra.ml.data import Pair

    base = [
        ("battery swap time", "the atlas robot swaps its battery in under thirty seconds"),
        ("fleet coordination", "warehouse fleets are coordinated by the conductor platform"),
        ("production languages", "approved production languages are rust python and typescript"),
        ("parental leave", "employees receive eighteen weeks of parental leave"),
        ("picking arm payload", "the picking arm handles items up to five kilograms"),
        ("headquarters location", "nimbus robotics is headquartered in portland oregon"),
    ]
    return [Pair(query=q, passage=p) for q, p in base * 4]


def test_train_bi_encoder_smoke(tmp_path, tiny_tokenizer):
    pytest.importorskip("torch")
    from orchestra.ml.train_biencoder import BiTrainConfig, train_bi_encoder
    from orchestra.ml.transformer import EncoderConfig

    pairs = _tiny_pairs()
    logs: list = []
    cfg = BiTrainConfig(epochs=1, batch_size=8, max_len=24, log_every=1)
    model = train_bi_encoder(
        pairs[:36], pairs[36:], tiny_tokenizer, cfg,
        encoder_cfg=EncoderConfig(vocab_size=tiny_tokenizer.vocab_size, max_len=24, dim=24, depth=1, heads=2),
        device="cpu", ckpt_path=tmp_path / "bi.pt", log=logs.append,
    )
    assert (tmp_path / "bi.pt").exists()
    emb = model.encode_texts(["battery swap time"], tiny_tokenizer, device="cpu", max_len=24)
    assert emb.shape == (1, 24)
    assert any("training complete" in m for m in logs)


def test_train_cross_encoder_smoke(tmp_path, tiny_tokenizer):
    pytest.importorskip("torch")
    from orchestra.ml.train_cross import CrossTrainConfig, train_cross_encoder
    from orchestra.ml.transformer import EncoderConfig

    pairs = _tiny_pairs()
    # Build (query, positive, label=1) / (query, wrong-passage, label=0) examples.
    examples = []
    for i, p in enumerate(pairs):
        neg = pairs[(i + 1) % len(pairs)].passage
        examples.append((p.query, p.passage, 1.0))
        examples.append((p.query, neg, 0.0))
    cfg = CrossTrainConfig(epochs=1, batch_size=8, max_len=32, log_every=1)
    model = train_cross_encoder(
        examples[:40], examples[40:], tiny_tokenizer, cfg,
        encoder_cfg=EncoderConfig(vocab_size=tiny_tokenizer.vocab_size, max_len=32, dim=24, depth=1, heads=2),
        device="cpu", ckpt_path=tmp_path / "cross.pt", log=lambda m: None,
    )
    assert (tmp_path / "cross.pt").exists()
    ranked = model.rerank(
        "battery swap time",
        [("a", "the atlas robot swaps its battery"), ("b", "unrelated text about leave")],
        tiny_tokenizer, device="cpu",
    )
    assert len(ranked) == 2 and ranked[0][1] >= ranked[1][1]


def test_evaluate_biencoder_with_reranker(tmp_path, tiny_tokenizer):
    pytest.importorskip("torch")
    from orchestra.ml.bi_encoder import BiEncoder
    from orchestra.ml.cross_encoder import CrossEncoder
    from orchestra.ml.eval import evaluate_biencoder
    from orchestra.ml.transformer import EncoderConfig

    cfg = EncoderConfig(vocab_size=tiny_tokenizer.vocab_size, max_len=24, dim=24, depth=1, heads=2)
    bi = BiEncoder(cfg)
    cross = CrossEncoder(EncoderConfig(vocab_size=tiny_tokenizer.vocab_size, max_len=32, dim=24, depth=1, heads=2))
    passages = [p.passage for p in _tiny_pairs()[:6]]
    queries = [p.query for p in _tiny_pairs()[:6]]
    gold = list(range(6))
    bi_m, rr_m = evaluate_biencoder(
        bi, tiny_tokenizer, queries, gold, passages, k=3, device="cpu",
        reranker=cross, rerank_pool=6,
    )
    assert 0.0 <= bi_m.recall <= 1.0 and 0.0 <= bi_m.ndcg <= 1.0
    assert rr_m is not None and 0.0 <= rr_m.recall <= 1.0
    assert "rerank" in rr_m.label


def test_ml_embedder_and_hnsw_when_checkpoint_present():
    """If a trained checkpoint exists, MLEmbedder + HNSWStore work end to end."""
    pytest.importorskip("torch")
    from orchestra.ml.adapters import HNSWStore, MLEmbedder, checkpoints_present

    if not checkpoints_present():
        pytest.skip("no trained checkpoint committed")
    emb = MLEmbedder()
    vecs = emb.embed(["battery swap", "parental leave policy"])
    assert vecs.shape[0] == 2 and vecs.shape[1] == emb.dimension
    store = HNSWStore()
    store.add(["a", "b"], vecs, ["battery swap text", "leave text"], [{}, {}])
    res = store.similarity_search(vecs[0], k=1)
    assert res and res[0].id == "a"
