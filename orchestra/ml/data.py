"""Training data: build (query, positive-passage) pairs.

Two sources, used together:

1. **MS-MARCO-mini** — a small slice auto-downloaded from a public mirror when
   the network is available (cached under ``artifacts/data/``). Each row gives a
   real query and its relevant passage.

2. **Synthetic corpus pairs** — generated from the project's own corpus by
   chunking documents and deriving natural queries from each chunk (title +
   salient sentences, question templating). This guarantees the pipeline works
   fully offline and that the model learns the project's domain.

We also provide a :func:`mine_hard_negatives` routine that uses a trained
bi-encoder to find the most confusable (but non-relevant) passages for each
query — the hard negatives the cross-encoder trains against.
"""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Pair",
    "build_synthetic_pairs",
    "load_msmarco_mini",
    "build_training_pairs",
    "train_val_split",
]

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on",
    "with", "as", "at", "by", "that", "this", "it", "its", "be", "from", "we",
    "our", "you", "your", "can", "will", "has", "have", "they", "their", "but",
    "not", "which", "when", "where", "how", "what", "who", "each", "per", "into",
}

# MS-MARCO BM25 triplets (query, positive passage, BM25 hard-negative passage)
# served as JSON rows by the public Hugging Face datasets-server (no auth, no
# `datasets` library needed). Paginated at 100 rows per request.
_HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=sentence-transformers%2Fmsmarco-bm25&config=triplet&split=train"
    "&offset={offset}&length={length}"
)


@dataclass
class Pair:
    query: str
    passage: str
    source: str = "synthetic"
    negative: str = ""  # optional gold hard-negative (MS-MARCO BM25 triplets)


# ---------------------------------------------------------------------------
# Synthetic pairs from the local corpus
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    text = re.sub(r"[#*`>_]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _keywords(text: str, k: int = 6) -> List[str]:
    counts: Dict[str, int] = {}
    for w in _WORD_RE.findall(text.lower()):
        if w in _STOP or len(w) < 3:
            continue
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:k]]


def _queries_for_passage(title: str, passage: str, rng: random.Random) -> List[str]:
    """Synthesize a few natural queries that the passage answers."""
    sents = [s for s in _SENT_RE.split(passage) if len(s.split()) >= 4]
    kws = _keywords(passage, k=8)
    out: List[str] = []
    if kws:
        out.append(" ".join(kws[: rng.randint(3, 5)]))
    if title:
        out.append(f"{title} {' '.join(kws[:3])}".strip())
    for s in sents[:2]:
        s_kw = _keywords(s, k=5)
        if s_kw:
            out.append(" ".join(s_kw))
    # Question-style templates over the top keywords.
    if kws:
        templates = [
            f"what is the {kws[0]}",
            f"how does {kws[0]} work",
            f"tell me about {kws[0]} {kws[1] if len(kws) > 1 else ''}".strip(),
        ]
        out.append(rng.choice(templates))
    # Dedup, drop empties.
    seen = set()
    uniq = []
    for q in out:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    return uniq[:4]


def build_synthetic_pairs(
    corpus_dir: str | Path,
    *,
    chunk_words: int = 60,
    overlap: int = 20,
    seed: int = 0,
) -> Tuple[List[Pair], List[str]]:
    """Build synthetic (query, passage) pairs from a corpus directory.

    Returns ``(pairs, passages)`` where ``passages`` is the deduplicated list of
    all passage texts (the retrieval corpus for evaluation).
    """
    rng = random.Random(seed)
    corpus_dir = Path(corpus_dir)
    files = sorted(p for p in corpus_dir.rglob("*") if p.suffix in {".md", ".txt"})
    passages: List[str] = []
    pairs: List[Pair] = []
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        title = ""
        for line in raw.splitlines():
            if line.strip().startswith("#"):
                title = _clean(line)
                break
        words = _clean(raw).split()
        step = max(1, chunk_words - overlap)
        for start in range(0, max(1, len(words)), step):
            chunk = " ".join(words[start : start + chunk_words])
            if len(chunk.split()) < 12:
                continue
            passages.append(chunk)
            for q in _queries_for_passage(title, chunk, rng):
                pairs.append(Pair(query=q, passage=chunk, source="synthetic"))
    # Dedup passages preserving order.
    seen = set()
    uniq_passages = []
    for p in passages:
        if p not in seen:
            seen.add(p)
            uniq_passages.append(p)
    return pairs, uniq_passages


# ---------------------------------------------------------------------------
# MS-MARCO mini
# ---------------------------------------------------------------------------
def _fetch_hf_rows(offset: int, length: int, timeout: float) -> list:
    # URL is a fixed https:// constant pointing at the public HF datasets-server;
    # only the integer offset/length are interpolated. No user-controlled scheme.
    url = _HF_ROWS_URL.format(offset=int(offset), length=int(length))
    req = urllib.request.Request(url, headers={"User-Agent": "orchestra-rag"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - fixed https URL
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return payload.get("rows", [])


def load_msmarco_mini(
    *,
    cache_dir: str | Path = "artifacts/data",
    limit: int = 800,
    timeout: float = 30.0,
) -> List[Pair]:
    """Download (and cache) a small MS-MARCO BM25-triplet slice.

    Each row yields a real ``(query, positive)`` pair plus a real BM25 hard
    ``negative``. Returns an empty list if the network is unavailable — callers
    must tolerate offline operation and fall back to synthetic pairs.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "msmarco_mini.jsonl"
    if cache.exists():
        rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) >= min(limit, 100):  # use cache if it has enough
            return [
                Pair(query=r["query"], passage=r["passage"], source="msmarco", negative=r.get("negative", ""))
                for r in rows[:limit]
            ]

    pairs: List[Pair] = []
    offset = 0
    page = 100  # datasets-server max page size
    try:
        while len(pairs) < limit:
            rows = _fetch_hf_rows(offset, min(page, limit - len(pairs)), timeout)
            if not rows:
                break
            for item in rows:
                r = item.get("row", {})
                q = (r.get("query") or "").strip()
                pos = (r.get("positive") or "").strip()
                neg = (r.get("negative") or "").strip()
                if q and pos and len(pos.split()) >= 5:
                    pairs.append(Pair(query=q, passage=pos, source="msmarco", negative=neg))
            offset += len(rows)
            if len(rows) < page:
                break
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        if not pairs:
            return []

    if pairs:
        cache.write_text(
            "\n".join(
                json.dumps({"query": p.query, "passage": p.passage, "negative": p.negative})
                for p in pairs
            ),
            encoding="utf-8",
        )
    return pairs[:limit]


def build_training_pairs(
    corpus_dir: str | Path,
    *,
    use_msmarco: bool = True,
    msmarco_limit: int = 800,
    seed: int = 0,
) -> Tuple[List[Pair], List[str]]:
    """Build the full training set: synthetic corpus pairs (+ MS-MARCO if available).

    Returns ``(pairs, eval_passages)`` where ``eval_passages`` is the corpus
    passage pool used by the evaluation harness.
    """
    pairs, passages = build_synthetic_pairs(corpus_dir, seed=seed)
    if use_msmarco:
        marco = load_msmarco_mini(limit=msmarco_limit)
        pairs = pairs + marco
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs, passages


def train_val_split(
    pairs: Sequence[Pair], *, val_frac: float = 0.15, seed: int = 0
) -> Tuple[List[Pair], List[Pair]]:
    """Deterministic train/val split."""
    items = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_frac))
    return items[n_val:], items[:n_val]


# ---------------------------------------------------------------------------
# Hard-negative mining (uses a trained bi-encoder)
# ---------------------------------------------------------------------------
def mine_hard_negatives(
    model,  # BiEncoder
    tokenizer,  # BPETokenizer
    queries: Sequence[str],
    positives: Sequence[str],
    passage_pool: Sequence[str],
    *,
    device: str = "cpu",
    n_neg: int = 2,
) -> List[Tuple[str, str, str]]:
    """Return ``(query, positive, hard_negative)`` triples.

    For each query, embed it and the passage pool, take the top similar passages,
    and pick the highest-ranked ones that are *not* the positive as hard negatives.
    """
    import torch  # local import keeps this module torch-free at import time

    pool = list(dict.fromkeys(passage_pool))  # dedup, keep order
    pool_emb = model.encode_texts(pool, tokenizer, device=device)
    q_emb = model.encode_texts(list(queries), tokenizer, device=device)
    sims = q_emb @ pool_emb.t()  # (Q, P)
    triples: List[Tuple[str, str, str]] = []
    pool_index = {p: i for i, p in enumerate(pool)}
    topk = torch.topk(sims, k=min(len(pool), n_neg + 5), dim=1).indices.tolist()
    for i, (q, pos) in enumerate(zip(queries, positives)):
        pos_idx = pool_index.get(pos, -1)
        negs: List[str] = []
        for j in topk[i]:
            if j == pos_idx:
                continue
            negs.append(pool[j])
            if len(negs) >= n_neg:
                break
        for neg in negs:
            triples.append((q, pos, neg))
    return triples


def msmarco_status(*, cache_dir: str | Path = "artifacts/data") -> Optional[int]:
    """Return the number of cached MS-MARCO rows, or ``None`` if not cached."""
    cache = Path(cache_dir) / "msmarco_mini.jsonl"
    if not cache.exists():
        return None
    return sum(1 for line in cache.read_text(encoding="utf-8").splitlines() if line.strip())
