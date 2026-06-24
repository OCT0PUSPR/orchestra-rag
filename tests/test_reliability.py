"""Tests for retries, budgets, and the circuit breaker."""

from __future__ import annotations

import pytest

from orchestra.reliability import (
    Budget,
    BudgetExceeded,
    CircuitBreaker,
    CircuitOpenError,
    estimate_cost_usd,
    estimate_tokens,
    with_retry,
)


def test_with_retry_succeeds_after_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    assert with_retry(flaky, attempts=5, base_delay=0.0, sleep=lambda _s: None) == "ok"
    assert calls["n"] == 3


def test_with_retry_reraises_after_exhaustion():
    def always_fail():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        with_retry(always_fail, attempts=2, base_delay=0.0, sleep=lambda _s: None)


def test_budget_rounds_enforced():
    b = Budget(max_rounds=2, max_cost_usd=10.0)
    b.charge_round()
    b.charge_round()
    with pytest.raises(BudgetExceeded):
        b.charge_round()


def test_budget_cost_enforced():
    b = Budget(max_rounds=10, max_cost_usd=0.01)
    b.charge_cost(0.005)
    with pytest.raises(BudgetExceeded):
        b.charge_cost(0.02)


def test_estimate_tokens_and_cost():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a b c d e") > 0
    assert estimate_cost_usd("mock", 1000, 1000) == 0.0
    assert estimate_cost_usd("anthropic", 1000, 1000) > 0.0


def test_circuit_breaker_opens_and_rejects():
    times = {"t": 0.0}
    cb = CircuitBreaker(fail_threshold=2, reset_after=10.0)
    cb._now = lambda: times["t"]  # type: ignore[assignment]

    def boom():
        raise ValueError("x")

    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(boom)
    # circuit now open -> fast reject
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: "should not run")
    # after reset window -> half-open, allows a trial
    times["t"] = 20.0
    assert cb.call(lambda: "recovered") == "recovered"
