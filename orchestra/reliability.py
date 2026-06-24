"""Reliability primitives: retries, budgets, and a circuit breaker.

* :func:`with_retry` wraps a callable with exponential-backoff retries. It uses
  ``tenacity`` if installed and otherwise falls back to a small built-in retry
  loop, so reliability does not hinge on an optional dependency.
* :class:`Budget` enforces per-query limits (max rounds and a max estimated
  USD cost) and raises :class:`BudgetExceeded` when crossed.
* :class:`CircuitBreaker` trips after consecutive failures to fail fast.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Type, TypeVar

__all__ = [
    "BudgetExceeded",
    "Budget",
    "CircuitBreaker",
    "CircuitOpenError",
    "with_retry",
    "estimate_tokens",
    "estimate_cost_usd",
]

T = TypeVar("T")


class BudgetExceeded(RuntimeError):
    """Raised when a query exceeds its round or cost budget."""


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit breaker is open."""


# --------------------------------------------------------------------------- #
# Retries                                                                      #
# --------------------------------------------------------------------------- #


def with_retry(
    fn: Callable[..., T],
    *args,
    attempts: int = 3,
    base_delay: float = 0.2,
    max_delay: float = 2.0,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
) -> T:
    """Call ``fn`` with exponential-backoff retries.

    Uses tenacity when available; otherwise a deterministic built-in loop.
    """
    try:
        import tenacity  # type: ignore

        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(attempts),
            wait=tenacity.wait_exponential(multiplier=base_delay, max=max_delay),
            retry=tenacity.retry_if_exception_type(retry_on),
            reraise=True,
        )
        return retryer(fn, *args, **kwargs)
    except ImportError:  # pragma: no cover - tenacity optional
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                return fn(*args, **kwargs)
            except retry_on as exc:  # type: ignore[misc]
                last_exc = exc
                if attempt == attempts - 1:
                    break
                sleep(min(base_delay * (2**attempt), max_delay))
        assert last_exc is not None
        raise last_exc


# --------------------------------------------------------------------------- #
# Budgets                                                                      #
# --------------------------------------------------------------------------- #

# Rough per-1k-token prices (USD) for estimation only. Not billing-grade.
_PRICE_PER_1K = {
    "anthropic": (0.005, 0.025),  # (input, output) ~ Opus-tier estimate
    "huggingface": (0.0002, 0.0002),
    "mock": (0.0, 0.0),
}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~1.3 tokens per whitespace word."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3) + 1


def estimate_cost_usd(backend: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate request cost in USD from token counts."""
    in_price, out_price = _PRICE_PER_1K.get(backend, (0.0, 0.0))
    return (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price


@dataclass
class Budget:
    """Per-query budget. Mutated as the query proceeds; raises when exceeded."""

    max_rounds: int = 3
    max_cost_usd: float = 1.0
    rounds_used: int = 0
    cost_used_usd: float = 0.0

    def charge_round(self) -> None:
        self.rounds_used += 1
        if self.rounds_used > self.max_rounds:
            raise BudgetExceeded(
                f"round budget exceeded: {self.rounds_used} > {self.max_rounds}"
            )

    def charge_cost(self, usd: float) -> None:
        self.cost_used_usd += usd
        if self.cost_used_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"cost budget exceeded: ${self.cost_used_usd:.4f} > ${self.max_cost_usd:.4f}"
            )

    def remaining_rounds(self) -> int:
        return max(0, self.max_rounds - self.rounds_used)


# --------------------------------------------------------------------------- #
# Circuit breaker                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class CircuitBreaker:
    """A minimal circuit breaker.

    After ``fail_threshold`` consecutive failures the circuit opens for
    ``reset_after`` seconds, during which calls are rejected fast. A success
    closes it again.
    """

    fail_threshold: int = 3
    reset_after: float = 15.0
    _failures: int = 0
    _opened_at: Optional[float] = field(default=None)
    _now: Callable[[], float] = time.monotonic

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if self._now() - self._opened_at >= self.reset_after:
            # half-open: allow a trial call
            return False
        return True

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        if self.is_open:
            raise CircuitOpenError("circuit breaker is open")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._failures += 1
            if self._failures >= self.fail_threshold:
                self._opened_at = self._now()
            raise
        # success
        self._failures = 0
        self._opened_at = None
        return result
