"""Pure, unit-testable text chunking.

The chunker splits a document into overlapping windows. It is "token-ish": it
counts whitespace-delimited words as a cheap, dependency-free proxy for tokens,
which is plenty for retrieval quality on prose and keeps the package free of a
heavyweight tokenizer dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["Chunk", "chunk_text", "chunk_words"]

_WORD_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    """A contiguous slice of a source document."""

    text: str
    index: int
    source: str = "unknown"
    start_word: int = 0
    end_word: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable identifier derived from source and position."""
        return f"{self.source}::{self.index}"


def _split_words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def chunk_words(
    words: List[str],
    chunk_size: int,
    overlap: int,
) -> List[range]:
    """Return the index ranges for each chunk over a word list.

    Separated from :func:`chunk_text` so the windowing math can be tested in
    isolation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not words:
        return []

    step = chunk_size - overlap
    ranges: List[range] = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        ranges.append(range(start, end))
        if end == n:
            break
        start += step
    return ranges


def chunk_text(
    text: str,
    *,
    chunk_size: int = 180,
    overlap: int = 40,
    source: str = "unknown",
    metadata: Optional[Dict[str, object]] = None,
) -> List[Chunk]:
    """Split ``text`` into overlapping :class:`Chunk` objects.

    Args:
        text: The raw document text.
        chunk_size: Target chunk length in words.
        overlap: Number of words shared between consecutive chunks.
        source: Identifier for the originating document (path or name).
        metadata: Extra metadata attached to every chunk.

    Returns:
        A list of chunks in document order. Empty input yields an empty list.
    """
    metadata = dict(metadata or {})
    words = _split_words(text)
    if not words:
        return []

    chunks: List[Chunk] = []
    for i, word_range in enumerate(chunk_words(words, chunk_size, overlap)):
        piece = " ".join(words[word_range.start : word_range.stop])
        chunks.append(
            Chunk(
                text=piece,
                index=i,
                source=source,
                start_word=word_range.start,
                end_word=word_range.stop,
                metadata=dict(metadata),
            )
        )
    return chunks
