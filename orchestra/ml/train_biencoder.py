"""Bi-encoder training loop: InfoNCE with in-batch negatives.

AdamW + linear-warmup / cosine-decay LR schedule, gradient clipping, periodic
validation, and best-checkpoint saving. Device auto-selects MPS > CUDA > CPU.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from orchestra.ml.bi_encoder import BiEncoder, info_nce_loss
from orchestra.ml.data import Pair
from orchestra.ml.device import select_device
from orchestra.ml.tokenizer import BPETokenizer
from orchestra.ml.transformer import EncoderConfig

__all__ = ["BiTrainConfig", "train_bi_encoder"]


@dataclass
class BiTrainConfig:
    epochs: int = 6
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    temperature: float = 0.05
    max_len: int = 96
    seed: int = 0
    log_every: int = 20


def _lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _iterate_batches(pairs: Sequence[Pair], batch_size: int, rng):
    order = list(range(len(pairs)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        if len(idx) < 2:  # InfoNCE needs >=2 for in-batch negatives
            continue
        yield [pairs[i] for i in idx]


def train_bi_encoder(
    train_pairs: Sequence[Pair],
    val_pairs: Sequence[Pair],
    tokenizer: BPETokenizer,
    cfg: BiTrainConfig,
    *,
    encoder_cfg: Optional[EncoderConfig] = None,
    device: Optional[str] = None,
    ckpt_path: Optional[str | Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> BiEncoder:
    """Train and return a bi-encoder. Logs real per-step / per-epoch metrics."""
    import random

    import torch

    log = log or print
    dev = select_device(device or "auto")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    enc_cfg = encoder_cfg or EncoderConfig(
        vocab_size=tokenizer.vocab_size, max_len=max(cfg.max_len, 96)
    )
    model = BiEncoder(enc_cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[bi] device={dev} params={n_params/1e6:.2f}M vocab={enc_cfg.vocab_size} "
        f"dim={enc_cfg.dim} depth={enc_cfg.depth} heads={enc_cfg.heads}")
    log(f"[bi] train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} "
        f"epochs={cfg.epochs} bs={cfg.batch_size} lr={cfg.lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_pairs) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * cfg.warmup_frac)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: _lr_lambda(s, total_steps, warmup)
    )

    def encode_batch_texts(texts: List[str]):
        ids, mask = tokenizer.encode_batch(texts, max_len=cfg.max_len)
        return (
            torch.tensor(ids, dtype=torch.long, device=dev),
            torch.tensor(mask, dtype=torch.long, device=dev),
        )

    def validate() -> float:
        model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch in _iterate_batches(val_pairs, cfg.batch_size, random.Random(0)):
                q_ids, q_mask = encode_batch_texts([p.query for p in batch])
                p_ids, p_mask = encode_batch_texts([p.passage for p in batch])
                q_emb = model(q_ids, q_mask)
                p_emb = model(p_ids, p_mask)
                losses.append(float(info_nce_loss(q_emb, p_emb, temperature=cfg.temperature).detach()))
        return sum(losses) / len(losses) if losses else float("nan")

    best_val = float("inf")
    step = 0
    start_time = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in _iterate_batches(train_pairs, cfg.batch_size, rng):
            q_ids, q_mask = encode_batch_texts([p.query for p in batch])
            p_ids, p_mask = encode_batch_texts([p.passage for p in batch])
            q_emb = model(q_ids, q_mask)
            p_emb = model(p_ids, p_mask)
            loss = info_nce_loss(q_emb, p_emb, temperature=cfg.temperature)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            running += float(loss.detach())
            seen += 1
            if step % cfg.log_every == 0:
                lr_now = sched.get_last_lr()[0]
                log(f"[bi] epoch {epoch} step {step}/{total_steps} "
                    f"loss={running/seen:.4f} lr={lr_now:.2e} "
                    f"elapsed={time.time()-start_time:.0f}s")
                running, seen = 0.0, 0
        val_loss = validate()
        log(f"[bi] == epoch {epoch} done val_loss={val_loss:.4f} ==")
        if val_loss < best_val and ckpt_path is not None:
            best_val = val_loss
            model.save(ckpt_path)
            log(f"[bi] saved checkpoint -> {ckpt_path} (val_loss={val_loss:.4f})")

    if ckpt_path is not None and best_val == float("inf"):
        model.save(ckpt_path)  # ensure something is saved even if val empty
    log(f"[bi] training complete in {time.time()-start_time:.0f}s best_val={best_val:.4f}")
    return model
