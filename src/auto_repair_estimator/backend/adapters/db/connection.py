from __future__ import annotations

import asyncpg
from loguru import logger

_pool: asyncpg.Pool | None = None


async def create_pool(dsn: str) -> asyncpg.Pool:
    global _pool
    logger.info("Creating asyncpg connection pool")
    _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
    logger.info("Connection pool created")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        logger.info("Closing asyncpg connection pool")
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialised; call create_pool() first")
    return _pool
