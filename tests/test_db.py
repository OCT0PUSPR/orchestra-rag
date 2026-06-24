"""Tests for the SQLAlchemy persistence layer and multi-tenant isolation."""

from __future__ import annotations

from orchestra.db import ApiKey, Collection, Database, Document, Tenant
from orchestra.security import generate_api_key, hash_api_key, key_prefix


def _db() -> Database:
    db = Database("sqlite:///:memory:")
    db.create_all()
    return db


def test_create_all_and_health():
    db = _db()
    assert db.healthy() is True


def test_tenant_apikey_collection_roundtrip():
    db = _db()
    with db.session() as s:
        t = Tenant(slug="acme", name="Acme Corp")
        s.add(t)
        s.flush()
        key = generate_api_key()
        s.add(ApiKey(tenant_id=t.id, key_hash=hash_api_key(key), prefix=key_prefix(key)))
        s.add(Collection(tenant_id=t.id, name="default"))

    with db.session() as s:
        from sqlalchemy import select

        tenant = s.scalar(select(Tenant).where(Tenant.slug == "acme"))
        assert tenant is not None
        assert len(tenant.api_keys) == 1
        assert len(tenant.collections) == 1


def test_multitenant_isolation():
    db = _db()
    with db.session() as s:
        from sqlalchemy import select

        for slug in ("t1", "t2"):
            t = Tenant(slug=slug, name=slug)
            s.add(t)
            s.flush()
            col = Collection(tenant_id=t.id, name="kb")
            s.add(col)
            s.flush()
            s.add(
                Document(
                    collection_id=col.id,
                    source=f"{slug}/doc.md",
                    content_hash=f"hash-{slug}",
                    num_chunks=3,
                )
            )

    with db.session() as s:
        from sqlalchemy import select

        t1 = s.scalar(select(Tenant).where(Tenant.slug == "t1"))
        t2 = s.scalar(select(Tenant).where(Tenant.slug == "t2"))
        docs1 = s.scalars(
            select(Document).join(Collection).where(Collection.tenant_id == t1.id)
        ).all()
        docs2 = s.scalars(
            select(Document).join(Collection).where(Collection.tenant_id == t2.id)
        ).all()
        # Each tenant sees only its own document.
        assert {d.source for d in docs1} == {"t1/doc.md"}
        assert {d.source for d in docs2} == {"t2/doc.md"}
