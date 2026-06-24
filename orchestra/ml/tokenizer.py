"""A small BPE tokenizer wrapper.

We use the ``tokenizers`` library for the BPE *algorithm only* (training the
merge table and the byte-level pre-tokenizer) — everything downstream
(transformer, pooling, losses, indexes) is implemented from scratch. This keeps
the from-scratch ML mandate intact while not re-implementing byte-level BPE
merge training, which is a solved, mechanical piece.

Special tokens:
    [PAD]=0  [UNK]=1  [CLS]=2  [SEP]=3  [MASK]=4

``encode_pair`` builds the cross-encoder input ``[CLS] q [SEP] p [SEP]`` with a
token-type ("segment") id per token, which the cross-encoder uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple

__all__ = ["BPETokenizer", "SPECIAL_TOKENS"]

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = 0, 1, 2, 3, 4


class BPETokenizer:
    """Byte-level BPE tokenizer with padding/truncation and segment ids.

    Thin wrapper over ``tokenizers.Tokenizer``; importing this module does not
    import torch.
    """

    def __init__(self, tokenizer: Any, vocab_size: int) -> None:
        self._tok: Any = tokenizer
        self.vocab_size = vocab_size

    # -- construction ------------------------------------------------------
    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        *,
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Train a byte-level BPE tokenizer on ``texts``."""
        from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
        from tokenizers.models import BPE

        tok = Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        tok.train_from_iterator(list(texts), trainer=trainer)
        return cls(tok, vocab_size=tok.get_vocab_size())

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        from tokenizers import Tokenizer

        path = Path(path)
        tok = Tokenizer.from_file(str(path))
        return cls(tok, vocab_size=tok.get_vocab_size())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))
        # Sidecar with vocab size for quick inspection.
        (path.parent / "tokenizer_meta.json").write_text(
            json.dumps({"vocab_size": self.vocab_size}), encoding="utf-8"
        )

    # -- encoding ----------------------------------------------------------
    def _ids(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def encode(self, text: str, *, max_len: int = 128, add_cls: bool = True) -> List[int]:
        """Encode a single text to ids with optional [CLS], truncated to ``max_len``."""
        ids = self._ids(text)
        if add_cls:
            ids = [CLS_ID] + ids
        return ids[:max_len]

    def encode_batch(
        self, texts: Sequence[str], *, max_len: int = 128, add_cls: bool = True
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Encode + pad a batch. Returns ``(input_ids, attention_mask)`` lists."""
        raw = [self.encode(t, max_len=max_len, add_cls=add_cls) for t in texts]
        width = max((len(r) for r in raw), default=1)
        width = max(1, min(width, max_len))
        input_ids: List[List[int]] = []
        masks: List[List[int]] = []
        for r in raw:
            r = r[:width]
            pad = width - len(r)
            input_ids.append(r + [PAD_ID] * pad)
            masks.append([1] * len(r) + [0] * pad)
        return input_ids, masks

    def encode_pair(
        self, query: str, passage: str, *, max_len: int = 192
    ) -> Tuple[List[int], List[int], List[int]]:
        """Encode ``[CLS] q [SEP] p [SEP]`` for the cross-encoder.

        Returns ``(input_ids, attention_mask, segment_ids)`` (single example,
        unpadded). The caller pads the batch.
        """
        q = self._ids(query)
        p = self._ids(passage)
        # Reserve room for 3 special tokens.
        budget = max_len - 3
        # Give the query at most ~1/3, passage the rest.
        q_max = max(8, budget // 3)
        q = q[:q_max]
        p = p[: budget - len(q)]
        ids = [CLS_ID] + q + [SEP_ID] + p + [SEP_ID]
        seg = [0] * (len(q) + 2) + [1] * (len(p) + 1)
        mask = [1] * len(ids)
        return ids, mask, seg

    @staticmethod
    def pad_pairs(
        rows: Sequence[Tuple[List[int], List[int], List[int]]],
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """Pad a batch of ``encode_pair`` outputs to a common width."""
        width = max((len(ids) for ids, _, _ in rows), default=1)
        out_ids, out_mask, out_seg = [], [], []
        for ids, mask, seg in rows:
            pad = width - len(ids)
            out_ids.append(ids + [PAD_ID] * pad)
            out_mask.append(mask + [0] * pad)
            out_seg.append(seg + [0] * pad)
        return out_ids, out_mask, out_seg
