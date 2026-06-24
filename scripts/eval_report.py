#!/usr/bin/env python3
"""Produce the honest, reproducible retrieval eval reported in the README.

Over a held-out MS-MARCO query set against the full passage pool, we score four
systems with recall@10 / nDCG@10:

  1. BM25 (from-scratch sparse)
  2. Bi-encoder (from-scratch dense, HNSW-equivalent exact ranking)
  3. Bi-encoder + cross-encoder rerank (top-50)
  4. Hybrid = RRF(BM25, bi-encoder)

Numbers are printed exactly as measured — no rounding-up, no cherry-picking.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from orchestra.ml.bi_encoder import BiEncoder  # noqa: E402
from orchestra.ml.cross_encoder import CrossEncoder  # noqa: E402
from orchestra.ml.data import build_synthetic_pairs, load_msmarco_mini  # noqa: E402
from orchestra.ml.device import select_device  # noqa: E402
from orchestra.ml.tokenizer import BPETokenizer  # noqa: E402
from orchestra.rag.hybrid import reciprocal_rank_fusion  # noqa: E402
from orchestra.rag.sparse import BM25Index  # noqa: E402

CKPT = Path(__file__).resolve().parent.parent / "orchestra" / "ml" / "checkpoints"
K = 10
RERANK_POOL = 50


def _metrics(ranked_lists, gold_ids):
    rec, ndcg = [], []
    for ranked, g in zip(ranked_lists, gold_ids):
        top = ranked[:K]
        hit = g in top
        rec.append(1.0 if hit else 0.0)
        ndcg.append(1.0 / math.log2(top.index(g) + 2) if hit else 0.0)
    n = len(gold_ids) or 1
    return sum(rec) / n, sum(ndcg) / n


def main() -> int:
    dev = str(select_device("auto"))
    corpus = str(Path(__file__).resolve().parent.parent / "data" / "sample_corpus")

    marco = load_msmarco_mini(limit=4000)
    pool, pidx = [], {}

    def add(t):
        if t not in pidx:
            pidx[t] = len(pool)
            pool.append(t)

    for p in marco:
        add(p.passage)
        if p.negative:
            add(p.negative)
    _, sp = build_synthetic_pairs(corpus, seed=0)
    for s in sp:
        add(s)

    rng = random.Random(0)
    idx = list(range(len(marco)))
    rng.shuffle(idx)
    ev = set(idx[: max(20, int(len(marco) * 0.15))])
    eq, eg = [], []
    for i, p in enumerate(marco):
        if i in ev:
            eq.append(p.query)
            eg.append(pidx[p.passage])

    print(f"held-out queries={len(eq)}  passage pool={len(pool)}  k={K}")

    bi = BiEncoder.load(CKPT / "bi_encoder.pt", map_location=dev).to(dev)
    tok = BPETokenizer.load(CKPT / "tokenizer.json")
    cross = CrossEncoder.load(CKPT / "cross_encoder.pt", map_location=dev).to(dev)

    # BM25
    bm = BM25Index()
    for i, t in enumerate(pool):
        bm.add(str(i), t)
    bm_ranked = []
    for q in eq:
        hits = bm.search(q, k=200)
        bm_ranked.append([int(h.doc_id) for h in hits])
    print(f"BM25                         recall@{K}={_metrics(bm_ranked, eg)[0]:.4f}  "
          f"nDCG@{K}={_metrics(bm_ranked, eg)[1]:.4f}")

    # Bi-encoder dense
    pe = bi.encode_texts(pool, tok, device=dev)
    qe = bi.encode_texts(eq, tok, device=dev)
    sims = qe @ pe.t()
    bi_ranked = torch.argsort(sims, dim=1, descending=True).tolist()
    print(f"Bi-encoder (dense)           recall@{K}={_metrics(bi_ranked, eg)[0]:.4f}  "
          f"nDCG@{K}={_metrics(bi_ranked, eg)[1]:.4f}")

    # Bi-encoder + cross rerank
    rr_ranked = []
    for i in range(len(eq)):
        cand = bi_ranked[i][:RERANK_POOL]
        scores = cross.score_pairs(eq[i], [pool[j] for j in cand], tok, device=dev)
        order = sorted(range(len(cand)), key=lambda x: scores[x], reverse=True)
        rr_ranked.append([cand[o] for o in order])
    print(f"Bi-encoder + cross-rerank    recall@{K}={_metrics(rr_ranked, eg)[0]:.4f}  "
          f"nDCG@{K}={_metrics(rr_ranked, eg)[1]:.4f}")

    # Hybrid RRF
    hy_ranked = []
    for i in range(len(eq)):
        dense_ids = [str(j) for j in bi_ranked[i][:200]]
        sparse_ids = [str(j) for j in bm_ranked[i][:200]]
        fused = reciprocal_rank_fusion(dense_ids, sparse_ids)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        hy_ranked.append([int(j) for j, _ in ordered])
    print(f"Hybrid RRF(BM25, bi-encoder) recall@{K}={_metrics(hy_ranked, eg)[0]:.4f}  "
          f"nDCG@{K}={_metrics(hy_ranked, eg)[1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
