"""Integration tests for the API via FastAPI TestClient (offline, MockLLM)."""

from __future__ import annotations

import io

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from orchestra.api.server import create_app  # noqa: E402


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["checks"]["vector_store"] is True


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"orchestra_" in r.content


def test_security_headers_present(client):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "X-Request-ID" in r.headers


def test_collections_seeded_public_default(client):
    r = client.get("/collections")
    assert r.status_code == 200
    names = {c["name"] for c in r.json()["collections"]}
    assert "default" in names
    default = next(c for c in r.json()["collections"] if c["name"] == "default")
    assert default["chunks"] > 0


def test_ask_returns_cited_answer(client):
    r = client.post(
        "/ask",
        json={"question": "How much parental leave do employees get?", "backend": "mock"},
    )
    assert r.status_code == 200
    text = r.text
    assert "final" in text
    assert "citations" in text
    # The streamed final event carries an inline citation marker.
    assert "[1]" in text or "[2]" in text


def test_upload_validation_rejects_exe(client):
    files = {"files": ("evil.exe", io.BytesIO(b"MZ\x00"), "application/x-msdownload")}
    r = client.post("/ingest", files=files)
    assert r.status_code == 400


def test_upload_accepts_markdown_and_indexes(client):
    content = b"# Secret Doc\nThe widget code is ZX-9000 and ships on Fridays."
    files = {"files": ("secret.md", io.BytesIO(content), "text/markdown")}
    r = client.post("/ingest?collection=mykb", files=files)
    assert r.status_code == 200
    assert r.json()["ingested_chunks"] >= 1

    # And it is retrievable in that collection.
    r2 = client.post(
        "/ask",
        json={"question": "What is the widget code?", "collection": "mykb", "backend": "mock"},
    )
    assert r2.status_code == 200


def test_ask_missing_collection_404(client):
    r = client.post(
        "/ask", json={"question": "anything", "collection": "nonexistent-xyz"}
    )
    assert r.status_code == 404


def test_create_and_delete_collection(client):
    r = client.post("/collections", json={"name": "temp"})
    assert r.status_code == 200
    r2 = client.delete("/collections/temp")
    assert r2.status_code == 200
    r3 = client.delete("/collections/temp")
    assert r3.status_code == 404


def test_invalid_collection_name_rejected(client):
    r = client.post("/collections", json={"name": "bad/name"})
    assert r.status_code == 400


def test_auth_required_when_enabled(monkeypatch):
    monkeypatch.setenv("OARAG_REQUIRE_AUTH", "true")
    app = create_app()
    c = TestClient(app)
    # No key -> 401 on a protected route.
    r = c.post("/ask", json={"question": "x"})
    assert r.status_code == 401
    # Health stays open.
    assert c.get("/health").status_code == 200


def test_multitenant_isolation_via_keys(monkeypatch, tmp_path):
    """Two valid keys map to two tenants; each only sees its own collection."""
    db_path = tmp_path / "tenants.sqlite"
    db_url = f"sqlite:///{db_path}"

    # Seed two tenants + keys into a shared on-disk DB BEFORE the app starts.
    from orchestra.db import ApiKey, Database, Tenant
    from orchestra.security import generate_api_key, hash_api_key, key_prefix

    db = Database(db_url)
    db.create_all()
    key_a = generate_api_key()
    key_b = generate_api_key()
    with db.session() as s:
        for slug, key in (("tenant-a", key_a), ("tenant-b", key_b)):
            t = Tenant(slug=slug, name=slug)
            s.add(t)
            s.flush()
            s.add(ApiKey(tenant_id=t.id, key_hash=hash_api_key(key), prefix=key_prefix(key)))

    monkeypatch.setenv("OARAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("OARAG_DATABASE_URL", db_url)
    app = create_app()
    c = TestClient(app)

    # Tenant A uploads a private doc into its own collection.
    secret = b"# A-only\nThe tenant-a passphrase is ALPHA-777."
    ra = c.post(
        "/ingest?collection=priv",
        files={"files": ("a.md", io.BytesIO(secret), "text/markdown")},
        headers={"X-API-Key": key_a},
    )
    assert ra.status_code == 200

    # Tenant B cannot see tenant A's collection at all.
    rb_list = c.get("/collections", headers={"X-API-Key": key_b})
    assert rb_list.status_code == 200
    b_names = {col["name"] for col in rb_list.json()["collections"]}
    assert "priv" not in b_names

    rb_ask = c.post(
        "/ask",
        json={"question": "passphrase?", "collection": "priv"},
        headers={"X-API-Key": key_b},
    )
    assert rb_ask.status_code == 404  # B's namespace has no 'priv' collection

    # An unknown key is rejected outright.
    assert c.post("/ask", json={"question": "x"}, headers={"X-API-Key": "bogus"}).status_code == 401
