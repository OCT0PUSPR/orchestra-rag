"""Unit tests for the pure chunker."""

from __future__ import annotations

import pytest

from orchestra.rag.chunking import chunk_text, chunk_words


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_single_chunk_when_short():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].index == 0


def test_chunking_with_overlap_covers_all_words():
    words = [f"w{i}" for i in range(100)]
    ranges = chunk_words(words, chunk_size=30, overlap=10)
    # Every word index must appear in at least one range.
    covered = set()
    for r in ranges:
        covered.update(range(r.start, r.stop))
    assert covered == set(range(100))


def test_overlap_is_respected():
    words = [f"w{i}" for i in range(100)]
    ranges = chunk_words(words, chunk_size=30, overlap=10)
    # step = chunk_size - overlap = 20
    assert ranges[0].start == 0
    assert ranges[1].start == 20
    # consecutive chunks share `overlap` words
    shared = set(range(ranges[0].start, ranges[0].stop)) & set(
        range(ranges[1].start, ranges[1].stop)
    )
    assert len(shared) == 10


def test_metadata_and_source_propagate():
    text = " ".join(f"token{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=20, overlap=5, source="doc.md", metadata={"k": "v"})
    assert all(c.source == "doc.md" for c in chunks)
    assert all(c.metadata["k"] == "v" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].id == "doc.md::0"


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_words(["a", "b"], chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_words(["a", "b"], chunk_size=5, overlap=5)
    with pytest.raises(ValueError):
        chunk_words(["a", "b"], chunk_size=5, overlap=-1)


def test_no_overlap_partitions_exactly():
    words = [f"w{i}" for i in range(60)]
    ranges = chunk_words(words, chunk_size=20, overlap=0)
    assert len(ranges) == 3
    assert ranges[0].start == 0 and ranges[1].start == 20 and ranges[2].start == 40
