"""Top-level pytest configuration.

The async event loop is fully owned by ``pytest-asyncio`` (``asyncio_mode = "auto"``
in ``pyproject.toml``). The legacy ``anyio_backend`` fixture was removed: the
``anyio`` plugin is disabled in ``addopts`` because it conflicted with
``pytest-asyncio`` over async-fixture lifecycle (asyncpg pools were being
created in one loop and consumed in another).
"""
