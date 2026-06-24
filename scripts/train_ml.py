#!/usr/bin/env python3
"""End-to-end training for the from-scratch retrieval models.

Pipeline (CPU/MPS, designed to finish in well under ~45 minutes on a laptop):

1. Build data: real MS-MARCO BM25 triplets (query, positive, hard-negative) —
   auto-downloaded + cached — plus synthetic (query, passage) pairs from the
   local project corpus. The union of all positives + negatives forms a realistic
   retrieval **passage pool** (hundreds–thousands of distinct passages).
2. Train a byte-level BPE tokenizer.
3. Train the bi-encoder with InfoNCE / in-batch negatives.
4. Train the cross-encoder reranker on the gold hard negatives (plus extra hard
   negatives mined by the trained bi-encoder).
5. Evaluate on a HELD-OUT query set against the full passage pool: recall@10 +
   nDCG@10, bi-encoder alone vs bi-encoder + cross-encoder reranking. Report the
   lift. Numbers are real — never fabricated.
6. Save checkpoints and export the bi-encoder to ONNX.

Usage:
    python scripts/train_ml.py                       # default run (uses MS-MARCO)
    python scripts/train_ml.py --no-msmarco          # offline, corpus-only
    python scripts/train_ml.py --msmarco-limit 3000 --epochs 8
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Make the repo importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestra.ml.bi_encoder import BiEncoder  # noqa: E402
from orchestra.ml.cross_encoder import CrossEncoder  # noqa: E402
from orchestra.ml.data import (  # noqa: E402
    Pair,
    build_synthetic_pairs,
    load_msmarco_mini,
)
from orchestra.ml.device import select_device  # noqa: E402
from orchestra.ml.eval import evaluate_biencoder  # noqa: E402
from orchestra.ml.tokenizer import BPETokenizer  # noqa: E402
from orchestra.ml.train_biencoder import BiTrainConfig, train_bi_encoder  # noqa: E402
from orchestra.ml.train_cross import CrossTrainConfig, train_cross_encoder  # noqa: E402
from orchestra.ml.transformer import EncoderConfig  # noqa: E402


def assemble(corpus_dir: str, msmarco_limit: int, use_msmarco: bool, seed: int):
    """Assemble training pairs + a passage pool with gold positives.

    Returns ``(train_pairs, eval_queries, eval_gold_ids, passage_pool)`` where
    every eval query's positive passage lives in ``passage_pool`` at index
    ``gold_id`` and is HELD OUT of training.
    """
    rng = random.Random(seed)

    marco: List[Pair] = []
    if use_msmarco:
        marco = load_msmarco_mini(limit=msmarco_limit)

    # Build the passage pool from all MS-MARCO positives + negatives (distinct).
    pool: List[str] = []
    pool_index = {}

    def add_passage(text: str) -> int:
        if text not in pool_index:
            pool_index[text] = len(pool)
            pool.append(text)
        return pool_index[text]

    for p in marco:
        add_passage(p.passage)
        if p.negative:
            add_passage(p.negative)

    # Synthetic corpus pairs + their passages (so the project domain is learned
    # and present in the pool).
    syn_pairs, syn_passages = build_synthetic_pairs(corpus_dir, seed=seed)
    for sp in syn_passages:
        add_passage(sp)

    # Hold out 15% of MS-MARCO queries for evaluation (their positives stay in
    # the pool, but the (query, positive) pair is removed from training).
    marco_idx = list(range(len(marco)))
    rng.shuffle(marco_idx)
    n_eval = max(20, int(len(marco) * 0.15)) if marco else 0
    eval_set = set(marco_idx[:n_eval])

    train_pairs: List[Pair] = []
    eval_queries: List[str] = []
    eval_gold: List[int] = []
    for i, p in enumerate(marco):
        if i in eval_set:
            eval_queries.append(p.query)
            eval_gold.append(pool_index[p.passage])
        else:
            train_pairs.append(p)

    # All synthetic pairs are training only.
    train_pairs.extend(syn_pairs)
    rng.shuffle(train_pairs)
    return train_pairs, eval_queries, eval_gold, pool


def build_cross_examples(
    train_pairs: List[Pair],
    bi: "BiEncoder",
    tokenizer: BPETokenizer,
    pool: List[str],
    device: str,
    *,
    neg_per_pos: int = 4,
    also_gold: bool = True,
) -> List[Tuple[str, str, float]]:
    """Build cross-encoder training examples.

    The reranker is applied at eval time to the *bi-encoder's own top candidates*,
    so we train it on exactly that distribution: for each training query we embed
    the query against the passage pool with the trained bi-encoder and take its
    top non-gold passages as **hard negatives** (`neg_per_pos` of them). The
    positive is the gold passage. This train/eval alignment is what makes the
    cross-encoder actually *lift* recall rather than scramble it.

    We additionally keep the gold MS-MARCO BM25 negatives (`also_gold`) for extra
    lexical-hard signal.
    """
    import torch

    examples: List[Tuple[str, str, float]] = []
    pidx = {p: i for i, p in enumerate(pool)}

    queries = [p.query for p in train_pairs]
    positives = [p.passage for p in train_pairs]

    # Mine bi-encoder hard negatives (the candidate distribution it will rerank).
    pool_emb = bi.encode_texts(pool, tokenizer, device=device)
    q_emb = bi.encode_texts(queries, tokenizer, device=device)
    sims = q_emb @ pool_emb.t()
    topk = torch.topk(sims, k=min(len(pool), neg_per_pos + 3), dim=1).indices.tolist()

    for i, (q, pos) in enumerate(zip(queries, positives)):
        gi = pidx.get(pos, -1)
        examples.append((q, pos, 1.0))
        added = 0
        for j in topk[i]:
            if j == gi:
                continue
            examples.append((q, pool[j], 0.0))
            added += 1
            if added >= neg_per_pos:
                break
        if also_gold and train_pairs[i].negative:
            examples.append((q, train_pairs[i].negative, 0.0))
    return examples


def main() -> int:
    ap = argparse.ArgumentParser(description="Train from-scratch retrieval models.")
    ap.add_argument("--corpus", default="data/sample_corpus")
    ap.add_argument("--out", default="orchestra/ml/checkpoints")
    ap.add_argument("--epochs", type=int, default=10, help="bi-encoder epochs")
    ap.add_argument("--cross-epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--dim", type=int, default=160)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--msmarco-limit", type=int, default=4000)
    ap.add_argument("--no-msmarco", action="store_true")
    ap.add_argument("--neg-per-pos", type=int, default=4, help="bi-encoder-mined hard negs per positive")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10, help="eval cutoff (recall@k / nDCG@k)")
    ap.add_argument("--rerank-pool", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(msg, flush=True)

    dev = str(select_device(args.device))

    # 1. Data ---------------------------------------------------------------
    log("=" * 72)
    log("STEP 1/7  Building data (MS-MARCO BM25 triplets + synthetic corpus)")
    train_pairs, eval_q, eval_gold, pool = assemble(
        args.corpus, args.msmarco_limit, not args.no_msmarco, args.seed
    )
    n_gold_neg = sum(1 for p in train_pairs if p.negative)
    log(f"  train_pairs={len(train_pairs)} (with gold hard-neg={n_gold_neg})")
    log(f"  passage pool={len(pool)}  held-out eval queries={len(eval_q)}")
    if not eval_q:
        log("  WARNING: no MS-MARCO eval queries (offline?). Using synthetic eval split.")
        from orchestra.ml.data import build_synthetic_pairs as _bsp

        syn, syn_pass = _bsp(args.corpus, seed=99)
        pidx = {p: i for i, p in enumerate(pool)}
        seen = set()
        for pr in syn:
            if pr.passage in pidx and pr.query.lower() not in seen:
                seen.add(pr.query.lower())
                eval_q.append(pr.query)
                eval_gold.append(pidx[pr.passage])

    # 2. Tokenizer ----------------------------------------------------------
    log("=" * 72)
    log("STEP 2/7  Training BPE tokenizer")
    corpus_texts = [p.query for p in train_pairs] + [p.passage for p in train_pairs] + pool + eval_q
    tokenizer = BPETokenizer.train(corpus_texts, vocab_size=args.vocab_size, min_frequency=1)
    tok_path = out / "tokenizer.json"
    tokenizer.save(tok_path)
    log(f"  vocab_size={tokenizer.vocab_size}  saved -> {tok_path}")

    # 3. Bi-encoder ---------------------------------------------------------
    log("=" * 72)
    log("STEP 3/7  Training bi-encoder (InfoNCE / in-batch negatives)")
    bi_val = train_pairs[: max(1, len(train_pairs) // 10)]
    bi_train = train_pairs[len(bi_val) :]
    bi_cfg = BiTrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, max_len=96, seed=args.seed
    )
    bi = train_bi_encoder(
        bi_train, bi_val, tokenizer, bi_cfg,
        encoder_cfg=EncoderConfig(
            vocab_size=tokenizer.vocab_size, max_len=96,
            dim=args.dim, depth=args.depth, heads=args.heads,
        ),
        device=args.device, ckpt_path=out / "bi_encoder.pt", log=log,
    )
    bi = BiEncoder.load(out / "bi_encoder.pt", map_location=dev).to(dev)

    # 4. Bi-encoder eval ----------------------------------------------------
    log("=" * 72)
    log("STEP 4/7  Evaluating bi-encoder on held-out queries")
    bi_metrics, _ = evaluate_biencoder(
        bi, tokenizer, eval_q, eval_gold, pool, k=args.k, device=dev
    )
    log("  " + bi_metrics.summary())

    # 5. Cross-encoder data -------------------------------------------------
    log("=" * 72)
    log("STEP 5/7  Building cross-encoder training data (gold + mined hard negs)")
    examples = build_cross_examples(
        train_pairs, bi, tokenizer, pool, dev, neg_per_pos=args.neg_per_pos
    )
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = max(1, len(examples) // 10)
    ce_val, ce_train = examples[:n_val], examples[n_val:]
    log(f"  cross examples={len(examples)} (train={len(ce_train)} val={len(ce_val)})")

    # 6. Cross-encoder ------------------------------------------------------
    log("=" * 72)
    log("STEP 6/7  Training cross-encoder reranker")
    cross_cfg = CrossTrainConfig(
        epochs=args.cross_epochs, batch_size=args.batch_size, max_len=160, seed=args.seed
    )
    train_cross_encoder(
        ce_train, ce_val, tokenizer, cross_cfg,
        encoder_cfg=EncoderConfig(
            vocab_size=tokenizer.vocab_size, max_len=160,
            dim=args.dim, depth=args.depth, heads=args.heads,
        ),
        device=args.device, ckpt_path=out / "cross_encoder.pt", log=log,
    )
    cross = CrossEncoder.load(out / "cross_encoder.pt", map_location=dev).to(dev)

    # 7. Final eval (bi vs +rerank) + ONNX ---------------------------------
    log("=" * 72)
    log("STEP 7/7  Final eval (bi-encoder vs +reranker) + ONNX export")
    bi_metrics, rr_metrics = evaluate_biencoder(
        bi, tokenizer, eval_q, eval_gold, pool,
        k=args.k, device=dev, reranker=cross, rerank_pool=args.rerank_pool,
    )
    log("  RESULT  " + bi_metrics.summary())
    if rr_metrics is not None:
        log("  RESULT  " + rr_metrics.summary())
        log(f"  LIFT    recall@{args.k} {rr_metrics.recall - bi_metrics.recall:+.4f}"
            f"   nDCG@{args.k} {rr_metrics.ndcg - bi_metrics.ndcg:+.4f}")

    try:
        from orchestra.ml.onnx_export import export_bi_encoder, verify_onnx

        onnx_path = out / "bi_encoder.onnx"
        export_bi_encoder(out / "bi_encoder.pt", tok_path, onnx_path)
        diff = verify_onnx(onnx_path, out / "bi_encoder.pt", tok_path)
        log(f"  ONNX exported -> {onnx_path}  (max|torch-onnx|={diff:.2e})")
    except Exception as exc:  # pragma: no cover - export is best-effort
        log(f"  ONNX export skipped: {exc}")

    for name in ["bi_encoder.pt", "cross_encoder.pt", "tokenizer.json", "bi_encoder.onnx"]:
        p = out / name
        if p.exists():
            log(f"  artifact {name}: {p.stat().st_size/1e6:.2f} MB")

    log("=" * 72)
    log(f"DONE in {time.time()-t0:.0f}s on device={dev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
