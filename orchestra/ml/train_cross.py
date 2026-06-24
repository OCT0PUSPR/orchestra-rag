"""Cross-encoder (reranker) training.

Trains on ``(query, positive)`` -> label 1 and ``(query, hard-negative)`` ->
label 0 pairs, where hard negatives are mined by the trained bi-encoder. Uses
binary cross-entropy with logits, AdamW, and warmup/cosine LR.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from orchestra.ml.cross_encoder import CrossEncoder
from orchestra.ml.device import select_device
from orchestra.ml.tokenizer import BPETokenizer
from orchestra.ml.transformer import EncoderConfig

__all__ = ["CrossTrainConfig", "train_cross_encoder"]

# A training example: (query, passage, label) with label in {0.0, 1.0}.
Example = Tuple[str, str, float]


@dataclass
class CrossTrainConfig:
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    max_len: int = 160
    seed: int = 0
    log_every: int = 10


def build_examples(triples: Sequence[Tuple[str, str, str]]) -> List[Example]:
    """Turn ``(query, positive, hard_negative)`` triples into labeled examples."""
    examples: List[Example] = []
    for q, pos, neg in triples:
        examples.append((q, pos, 1.0))
        examples.append((q, neg, 0.0))
    return examples


def _lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_cross_encoder(
    train_examples: Sequence[Example],
    val_examples: Sequence[Example],
    tokenizer: BPETokenizer,
    cfg: CrossTrainConfig,
    *,
    encoder_cfg: Optional[EncoderConfig] = None,
    device: Optional[str] = None,
    ckpt_path: Optional[str | Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> CrossEncoder:
    """Train and return a cross-encoder reranker."""
    import random

    import torch
    import torch.nn.functional as F

    log = log or print
    dev = select_device(device or "auto")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    enc_cfg = encoder_cfg or EncoderConfig(
        vocab_size=tokenizer.vocab_size, max_len=max(cfg.max_len, 160)
    )
    model = CrossEncoder(enc_cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[cross] device={dev} params={n_params/1e6:.2f}M "
        f"dim={enc_cfg.dim} depth={enc_cfg.depth} heads={enc_cfg.heads}")
    log(f"[cross] train_ex={len(train_examples)} val_ex={len(val_examples)} "
        f"epochs={cfg.epochs} bs={cfg.batch_size} lr={cfg.lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_examples) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * cfg.warmup_frac)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: _lr_lambda(s, total_steps, warmup)
    )

    def encode(batch: List[Example]):
        rows = [tokenizer.encode_pair(q, p, max_len=cfg.max_len) for q, p, _ in batch]
        ids, mask, seg = tokenizer.pad_pairs(rows)
        labels = [lbl for _, _, lbl in batch]
        return (
            torch.tensor(ids, dtype=torch.long, device=dev),
            torch.tensor(mask, dtype=torch.long, device=dev),
            torch.tensor(seg, dtype=torch.long, device=dev),
            torch.tensor(labels, dtype=torch.float32, device=dev),
        )

    def batches(data: Sequence[Example], shuffle_rng):
        order = list(range(len(data)))
        shuffle_rng.shuffle(order)
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            yield [data[i] for i in idx]

    def validate() -> Tuple[float, float]:
        model.eval()
        losses: List[float] = []
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in batches(val_examples, random.Random(0)):
                ids, mask, seg, labels = encode(batch)
                logits = model(ids, mask, seg)
                losses.append(float(F.binary_cross_entropy_with_logits(logits, labels)))
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += int((preds == labels).sum())
                total += len(labels)
        loss = sum(losses) / len(losses) if losses else float("nan")
        acc = correct / total if total else float("nan")
        return loss, acc

    best_val = float("inf")
    step = 0
    start_time = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in batches(train_examples, rng):
            ids, mask, seg, labels = encode(batch)
            logits = model(ids, mask, seg)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            running += float(loss.detach())
            seen += 1
            if step % cfg.log_every == 0:
                log(f"[cross] epoch {epoch} step {step}/{total_steps} "
                    f"loss={running/seen:.4f} lr={sched.get_last_lr()[0]:.2e} "
                    f"elapsed={time.time()-start_time:.0f}s")
                running, seen = 0.0, 0
        val_loss, val_acc = validate()
        log(f"[cross] == epoch {epoch} done val_loss={val_loss:.4f} val_acc={val_acc:.3f} ==")
        if val_loss < best_val and ckpt_path is not None:
            best_val = val_loss
            model.save(ckpt_path)
            log(f"[cross] saved checkpoint -> {ckpt_path} (val_loss={val_loss:.4f})")

    if ckpt_path is not None and best_val == float("inf"):
        model.save(ckpt_path)
    log(f"[cross] training complete in {time.time()-start_time:.0f}s best_val={best_val:.4f}")
    return model
