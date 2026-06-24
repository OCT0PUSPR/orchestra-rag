"""Locust load test for the orchestra-rag API.

Run against a running server::

    locust -f locustfile.py --host http://localhost:8000

Exercises /health, /ready, and the SSE /ask endpoint under load.
"""

from __future__ import annotations

import random

try:
    from locust import HttpUser, between, task
except ImportError:  # pragma: no cover - locust optional
    HttpUser = object  # type: ignore

    def task(*_a, **_k):  # type: ignore
        def deco(f):
            return f

        return deco

    def between(*_a, **_k):  # type: ignore
        return 0


_QUESTIONS = [
    "How long does the Atlas-7 battery last?",
    "What programming languages are approved for production?",
    "How much parental leave do employees get?",
    "How does Conductor prevent collisions?",
    "Where is Nimbus headquartered?",
]


class OrchestraUser(HttpUser):  # type: ignore[misc]
    wait_time = between(1, 3)

    @task(3)
    def health(self):
        self.client.get("/health")

    @task(1)
    def ready(self):
        self.client.get("/ready")

    @task(5)
    def ask(self):
        question = random.choice(_QUESTIONS)
        # Stream the SSE response to completion.
        with self.client.post(
            "/ask",
            json={"question": question, "backend": "mock"},
            stream=True,
            catch_response=True,
            name="/ask",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")
                return
            saw_final = any(b"\"type\": \"final\"" in chunk for chunk in resp.iter_lines())
            if saw_final:
                resp.success()
            else:
                resp.failure("no final event")
