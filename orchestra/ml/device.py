"""Device selection for training/inference: MPS > CUDA > CPU.

Kept tiny and torch-guarded so importing it without torch raises a clear error
only when actually used.
"""

from __future__ import annotations

from typing import Any

__all__ = ["select_device", "device_name"]


def select_device(prefer: str = "auto") -> Any:
    """Return a ``torch.device``, preferring Apple MPS, then CUDA, then CPU.

    Args:
        prefer: ``"auto"`` (default), or force one of ``"mps" | "cuda" | "cpu"``.
    """
    import torch

    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(prefer: str = "auto") -> str:
    """A human-readable name for the selected device (e.g. ``"mps"``)."""
    return str(select_device(prefer))
