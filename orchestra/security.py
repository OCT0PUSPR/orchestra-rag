"""Security primitives: API keys, rate limiting, and upload validation.

* API keys are random tokens; only a SHA-256 hash and a short prefix are stored.
  Verification is constant-time.
* A token-bucket rate limiter caps requests per key per minute (in-memory; swap
  for Redis in a multi-process deployment).
* Upload validation enforces size, count, extension, and MIME allowlists and
  sanitizes filenames against path traversal.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Dict, Optional, Tuple

__all__ = [
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    "key_prefix",
    "RateLimiter",
    "sanitize_filename",
    "validate_upload",
    "UploadValidationError",
    "ALLOWED_EXTENSIONS",
    "ALLOWED_MIME_TYPES",
]

_KEY_PREFIX = "oarag_"
ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}
ALLOWED_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "text/html",
    "application/pdf",
    "application/octet-stream",  # browsers sometimes send this for .md
    "",  # missing content-type; extension still enforced
}

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- #
# API keys                                                                     #
# --------------------------------------------------------------------------- #


def generate_api_key() -> str:
    """Return a new random API key (shown to the user exactly once)."""
    return _KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """SHA-256 hash of an API key (what we store)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_prefix(key: str) -> str:
    """A short, non-secret prefix for display/lookup hints."""
    return key[:12]


def verify_api_key(key: str, stored_hash: str) -> bool:
    """Constant-time verification of a key against its stored hash."""
    return hmac.compare_digest(hash_api_key(key), stored_hash)


# --------------------------------------------------------------------------- #
# Rate limiting                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class RateLimiter:
    """In-memory token-bucket rate limiter, keyed by an arbitrary identity."""

    per_minute: int = 60
    _buckets: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    _now = staticmethod(time.monotonic)

    def allow(self, identity: str) -> bool:
        """Return True if a request from ``identity`` is allowed right now."""
        rate = self.per_minute / 60.0
        capacity = float(self.per_minute)
        now = self._now()
        tokens, last = self._buckets.get(identity, (capacity, now))
        tokens = min(capacity, tokens + (now - last) * rate)
        if tokens < 1.0:
            self._buckets[identity] = (tokens, now)
            return False
        self._buckets[identity] = (tokens - 1.0, now)
        return True


# --------------------------------------------------------------------------- #
# Upload validation                                                           #
# --------------------------------------------------------------------------- #


class UploadValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def sanitize_filename(name: str) -> str:
    """Return a safe basename: strip directories, traversal, and unsafe chars."""
    # Drop any path components and traversal.
    base = PurePosixPath(name.replace("\\", "/")).name
    base = base.replace("..", "")
    base = _SAFE_NAME_RE.sub("_", base).strip("._")
    if not base:
        base = "upload.txt"
    return base[:255]


def validate_upload(
    filename: str,
    content: bytes,
    content_type: Optional[str] = None,
    *,
    max_bytes: int = 10 * 1024 * 1024,
) -> str:
    """Validate one uploaded file; return the sanitized filename or raise.

    Checks size, extension allowlist, and MIME allowlist; sanitizes the name.
    """
    if len(content) == 0:
        raise UploadValidationError("empty file")
    if len(content) > max_bytes:
        raise UploadValidationError(
            f"file too large: {len(content)} bytes > {max_bytes} limit"
        )
    safe = sanitize_filename(filename)
    suffix = PurePosixPath(safe).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(f"unsupported extension: {suffix or '(none)'}")
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct not in ALLOWED_MIME_TYPES:
        raise UploadValidationError(f"unsupported content-type: {ct}")
    return safe
