"""Tests for API keys, rate limiting, and upload validation."""

from __future__ import annotations

import pytest

from orchestra.security import (
    RateLimiter,
    UploadValidationError,
    generate_api_key,
    hash_api_key,
    key_prefix,
    sanitize_filename,
    validate_upload,
    verify_api_key,
)


def test_api_key_roundtrip():
    key = generate_api_key()
    assert key.startswith("oarag_")
    h = hash_api_key(key)
    assert verify_api_key(key, h) is True
    assert verify_api_key("wrong", h) is False
    assert key_prefix(key) == key[:12]


def test_api_key_hash_is_not_the_key():
    key = generate_api_key()
    assert hash_api_key(key) != key
    assert len(hash_api_key(key)) == 64  # sha256 hex


def test_rate_limiter_enforces_limit():
    rl = RateLimiter(per_minute=3)
    results = [rl.allow("k") for _ in range(5)]
    assert results[:3] == [True, True, True]
    assert results[3:] == [False, False]


def test_rate_limiter_is_per_identity():
    rl = RateLimiter(per_minute=1)
    assert rl.allow("a") is True
    assert rl.allow("b") is True  # separate bucket
    assert rl.allow("a") is False


def test_sanitize_filename_blocks_traversal():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("a/b/c.md") == "c.md"
    assert sanitize_filename("..\\..\\win.txt") == "win.txt"
    assert sanitize_filename("") == "upload.txt"


def test_validate_upload_accepts_allowed():
    assert validate_upload("notes.md", b"# hi", "text/markdown") == "notes.md"
    assert validate_upload("page.html", b"<p>x</p>", "text/html") == "page.html"
    assert validate_upload("doc.txt", b"x", "") == "doc.txt"  # missing CT, ok by ext


def test_validate_upload_rejects_bad():
    with pytest.raises(UploadValidationError):
        validate_upload("evil.exe", b"MZ", "application/x-msdownload")
    with pytest.raises(UploadValidationError):
        validate_upload("empty.txt", b"", "text/plain")
    with pytest.raises(UploadValidationError):
        validate_upload("big.txt", b"x" * 200, "text/plain", max_bytes=100)
