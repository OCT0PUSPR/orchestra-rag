"""Tests for document loaders (txt/md/html, directory ingest, sanitization)."""

from __future__ import annotations

from orchestra.rag.loaders import Document, load_path, load_paths


def test_load_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world", encoding="utf-8")
    docs = load_path(p)
    assert len(docs) == 1
    assert docs[0].text == "hello world"
    assert docs[0].source == str(p)


def test_load_markdown(tmp_path):
    p = tmp_path / "b.md"
    p.write_text("# Title\n\nbody text", encoding="utf-8")
    docs = load_path(p)
    assert len(docs) == 1
    assert "body text" in docs[0].text


def test_load_html_strips_tags_and_scripts(tmp_path):
    p = tmp_path / "c.html"
    p.write_text(
        "<html><head><style>x{}</style></head><body><script>bad()</script>"
        "<p>visible content</p></body></html>",
        encoding="utf-8",
    )
    docs = load_path(p)
    assert len(docs) == 1
    assert "visible content" in docs[0].text
    assert "bad()" not in docs[0].text
    assert "x{}" not in docs[0].text


def test_directory_ingest(tmp_path):
    (tmp_path / "one.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "two.md").write_text("beta", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01")
    docs = load_path(tmp_path)
    sources = {d.text for d in docs}
    assert "alpha" in sources and "beta" in sources
    assert all(d.text not in ("",) for d in docs)


def test_unsupported_and_missing(tmp_path):
    assert load_path(tmp_path / "missing.txt") == []
    bad = tmp_path / "x.bin"
    bad.write_bytes(b"\x00")
    assert load_path(bad) == []


def test_empty_file_skipped(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n  ", encoding="utf-8")
    assert load_path(p) == []


def test_load_paths_multiple(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("one", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("two", encoding="utf-8")
    docs = load_paths([a, b])
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
