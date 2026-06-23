"""Document loaders for .txt / .md / .html / .pdf and directory ingestion.

PDF and HTML parsing degrade gracefully: if the optional dependency is missing,
PDFs are skipped with a warning and HTML falls back to a built-in tag stripper.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger("orchestra.rag.loaders")

__all__ = ["Document", "load_path", "load_paths", "SUPPORTED_SUFFIXES"]

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


@dataclass
class Document:
    """A loaded document: its full text plus its source identifier."""

    text: str
    source: str


class _TextExtractingParser(HTMLParser):
    """Collect human-readable text from HTML, dropping script/style content."""

    _SKIP = {"script", "style", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:  # noqa: D401
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def text(self) -> str:
        return "\n".join(self._parts)


def _strip_html(raw: str) -> str:
    parser = _TextExtractingParser()
    parser.feed(raw)
    parser.close()
    text = parser.text()
    # Collapse runaway whitespace.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _load_pdf(path: Path) -> Optional[str]:
    try:
        import pypdf  # type: ignore
    except ImportError:
        logger.warning(
            "pypdf is not installed; skipping PDF %s. Install with `pip install pypdf`.",
            path,
        )
        return None
    try:
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    except Exception as exc:  # pragma: no cover - depends on file contents
        logger.warning("Failed to read PDF %s: %s", path, exc)
        return None


def load_path(path: str | Path) -> List[Document]:
    """Load a single file or every supported file under a directory.

    Returns a list of :class:`Document`. Unsupported or unreadable files are
    skipped with a warning rather than raising.
    """
    p = Path(path)
    if p.is_dir():
        return _load_directory(p)
    doc = _load_file(p)
    return [doc] if doc is not None else []


def _load_directory(directory: Path) -> List[Document]:
    docs: List[Document] = []
    for child in sorted(directory.rglob("*")):
        if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
            doc = _load_file(child)
            if doc is not None:
                docs.append(doc)
    if not docs:
        logger.warning("No supported documents found under %s", directory)
    return docs


def _load_file(path: Path) -> Optional[Document]:
    if not path.exists():
        logger.warning("Path does not exist: %s", path)
        return None
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        logger.warning("Unsupported file type %s for %s", suffix, path)
        return None

    if suffix == ".pdf":
        text = _load_pdf(path)
    elif suffix in {".html", ".htm"}:
        text = _strip_html(path.read_text(encoding="utf-8", errors="replace"))
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    if not text or not text.strip():
        logger.warning("No extractable text in %s", path)
        return None
    return Document(text=text, source=str(path))


def load_paths(paths: Sequence[str | Path] | Iterable[str | Path]) -> List[Document]:
    """Load every path in ``paths`` (files or directories), concatenated."""
    docs: List[Document] = []
    for path in paths:
        docs.extend(load_path(path))
    return docs
