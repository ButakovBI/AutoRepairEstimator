"""Integration-test fixtures that wire up a real PostgreSQL database.

Two strategies are supported (checked in order):

1. **Environment variable** ``INTEGRATION_DB_URL``: Set this to an asyncpg DSN
   (``postgresql://user:pass@host:port/db``) to point at an already-running
   PostgreSQL instance — for example one started via ``docker-compose up postgres``.

2. **Testcontainers** (requires Docker): If the env var is absent, the fixture
   tries to spin up a ``postgres:16-alpine`` container automatically.

If neither is available (no env var, Docker not accessible) all tests that
depend on ``postgres_dsn`` are automatically **skipped** so the fast unit-test
suite continues to pass without any infrastructure.

The ``api_client`` fixture creates a FastAPI app wired to real Postgres repos
and returns an ``httpx.AsyncClient`` for HTTP-level assertions.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import re

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

_INIT_SQL_PATH = pathlib.Path(__file__).parents[2] / "docker" / "init.sql"

# Tables cleared before each test (order matters due to FK constraints).
_TRUNCATE_TABLES = [
    "detected_damages",
    "detected_parts",
    "outbox_events",
    "repair_requests",
]


async def _truncate_all_tables(dsn: str) -> None:
    """Best-effort cleanup between integration tests.

    Uses a *fresh* connection instead of the test pool so teardown remains
    stable even if a pooled connection was dropped by Postgres mid-test.
    """
    conn: asyncpg.Connection | None = None
    try:
        conn = await asyncpg.connect(dsn)
        for table in _TRUNCATE_TABLES:
            await conn.execute(f"TRUNCATE {table} CASCADE")
    finally:
        if conn is not None:
            await conn.close()


async def _init_schema(dsn: str) -> None:
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    sql = _INIT_SQL_PATH.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        await conn.execute(sql)
    await pool.close()


def _try_testcontainers() -> str | None:
    """Return an asyncpg DSN from a fresh Postgres container, or None if Docker is unavailable."""
    try:
        from testcontainers.postgres import PostgresContainer  # noqa: PLC0415

        container = PostgresContainer("postgres:16-alpine")
        container.start()
        raw = container.get_connection_url()
        dsn = re.sub(r"^postgresql\+\w+://", "postgresql://", raw)
        asyncio.run(_init_schema(dsn))
        # Store the container on the fixture so it's stopped on teardown.
        _try_testcontainers._container = container  # type: ignore[attr-defined]
        return dsn
    except Exception:
        return None


@pytest.fixture(scope="session")
def postgres_dsn() -> str:  # type: ignore[return]
    """Provide a Postgres DSN for integration tests.

    Resolution order:
    1. ``INTEGRATION_DB_URL`` environment variable (pre-existing DB).
    2. Testcontainers (auto-started Docker container).
    3. ``pytest.skip`` if neither is available.
    """
    # Strategy 1 — explicit env var (e.g. docker-compose postgres)
    env_dsn = os.getenv("INTEGRATION_DB_URL")
    if env_dsn:
        asyncio.run(_init_schema(env_dsn))
        yield env_dsn
        return

    # Strategy 2 — testcontainers
    dsn = _try_testcontainers()
    if dsn is not None:
        yield dsn
        container = getattr(_try_testcontainers, "_container", None)
        if container is not None:
            container.stop()
        return

    # Strategy 3 — nothing available, skip all DB tests
    pytest.skip("No PostgreSQL available (set INTEGRATION_DB_URL or start Docker)", allow_module_level=True)


@pytest.fixture
async def db_pool(postgres_dsn: str) -> asyncpg.Pool:  # type: ignore[return]
    """Function-scoped pool; truncates all tables after each test."""
    pool: asyncpg.Pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=3)
    try:
        yield pool
    finally:
        # Close test pool first; cleanup then runs through a fresh
        # standalone connection to avoid teardown flakes when a pooled
        # connection gets severed by the DB engine.
        try:
            await pool.close()
        except Exception:
            # If the pool is already broken/closed, cleanup still proceeds
            # through a fresh standalone connection below.
            pass
        await _truncate_all_tables(postgres_dsn)


@pytest.fixture
async def api_client(db_pool: asyncpg.Pool) -> AsyncClient:  # type: ignore[return]
    """FastAPI test client backed by real PostgreSQL repositories.

    This is the primary fixture for controller+database integration tests.
    The app is built fresh for each test so app.state is isolated.
    """
    from auto_repair_estimator.backend.adapters.repositories.postgres_damage_repository import PostgresDamageRepository
    from auto_repair_estimator.backend.adapters.repositories.postgres_outbox_repository import PostgresOutboxRepository
    from auto_repair_estimator.backend.adapters.repositories.postgres_pricing_rule_repository import (
        PostgresPricingRuleRepository,
    )
    from auto_repair_estimator.backend.adapters.repositories.postgres_repair_request_repository import (
        PostgresRepairRequestRepository,
    )
    from auto_repair_estimator.backend.main import create_app

    app = create_app()
    app.state.request_repo = PostgresRepairRequestRepository(db_pool)
    app.state.damage_repo = PostgresDamageRepository(db_pool)
    app.state.pricing_rule_repo = PostgresPricingRuleRepository(db_pool)
    app.state.outbox_repo = PostgresOutboxRepository(db_pool)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
