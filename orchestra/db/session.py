"""Database session factory and a thin :class:`Database` helper.

Defaults to SQLite. ``create_all`` is provided for tests and dev; production
uses Alembic migrations (see ``alembic/``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from orchestra.db.models import Base


class Database:
    """Owns the engine + session factory for one database URL."""

    def __init__(self, url: str = "sqlite:///./orchestra.sqlite", echo: bool = False) -> None:
        self.url = url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, echo=echo, future=True, connect_args=connect_args)
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)

    def create_all(self) -> None:
        """Create tables directly (tests/dev). Production uses Alembic."""
        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        sess = self._Session()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    def healthy(self) -> bool:
        """Lightweight connectivity check for /ready."""
        from sqlalchemy import text

        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


_default: Optional[Database] = None


def get_database(url: Optional[str] = None) -> Database:
    """Return a process-wide default Database (or a new one for ``url``)."""
    global _default
    if url is not None:
        return Database(url)
    if _default is None:
        from orchestra.config import load_settings

        _default = Database(load_settings().database_url)
    return _default
