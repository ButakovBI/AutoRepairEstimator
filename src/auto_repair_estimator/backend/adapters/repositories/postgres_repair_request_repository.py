from __future__ import annotations

from datetime import UTC, datetime

import asyncpg
from loguru import logger

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus


class PostgresRepairRequestRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def add(self, request: RepairRequest) -> None:
        logger.debug("INSERT repair_request id={}", request.id)
        await self._pool.execute(
            """
            INSERT INTO repair_requests
                (id, chat_id, user_id, mode, status,
                 created_at, updated_at, timeout_at,
                 original_image_key, composited_image_key,
                 ml_error_code, ml_error_message, idempotency_key)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            """,
            request.id,
            request.chat_id,
            request.user_id,
            request.mode.value,
            request.status.value,
            request.created_at,
            request.updated_at,
            request.timeout_at,
            request.original_image_key,
            request.composited_image_key,
            request.ml_error_code,
            request.ml_error_message,
            request.idempotency_key,
        )

    async def get(self, request_id: str) -> RepairRequest | None:
        row = await self._pool.fetchrow("SELECT * FROM repair_requests WHERE id = $1", request_id)
        if row is None:
            return None
        return self._from_row(row)

    async def get_by_idempotency_key(self, idempotency_key: str) -> RepairRequest | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM repair_requests WHERE idempotency_key = $1",
            idempotency_key,
        )
        if row is None:
            return None
        return self._from_row(row)

    async def update(self, request: RepairRequest) -> None:
        logger.debug("UPDATE repair_request id={} status={}", request.id, request.status)
        await self._pool.execute(
            """
            UPDATE repair_requests
            SET status=$2, updated_at=$3, original_image_key=$4,
                composited_image_key=$5, timeout_at=$6,
                ml_error_code=$7, ml_error_message=$8
            WHERE id=$1
            """,
            request.id,
            request.status.value,
            request.updated_at,
            request.original_image_key,
            request.composited_image_key,
            request.timeout_at,
            request.ml_error_code,
            request.ml_error_message,
        )

    async def get_timed_out_requests(self) -> list[RepairRequest]:
        terminal = ("done", "failed")
        rows = await self._pool.fetch(
            """
            SELECT * FROM repair_requests
            WHERE status != ALL($1::text[]) AND timeout_at < $2
            """,
            list(terminal),
            datetime.now(UTC),
        )
        return [self._from_row(r) for r in rows]

    async def get_latest_active_by_chat_id(self, chat_id: int) -> RepairRequest | None:
        # We intentionally pick the *latest* non-terminal request so a user
        # who accidentally created two overlapping sessions always sees
        # their newest one. The index idx_repair_requests_status_timeout
        # doesn't cover chat_id, but with a small non-terminal working set
        # per user this is a cheap scan.
        terminal = ("done", "failed")
        row = await self._pool.fetchrow(
            """
            SELECT * FROM repair_requests
            WHERE chat_id = $1 AND status != ALL($2::text[])
            ORDER BY created_at DESC
            LIMIT 1
            """,
            chat_id,
            list(terminal),
        )
        if row is None:
            return None
        return self._from_row(row)

    @staticmethod
    def _from_row(row: asyncpg.Record) -> RepairRequest:
        # ``.get`` on an asyncpg.Record falls back to the supplied
        # default when the column is absent, so old deployments that
        # haven't run the ``ADD COLUMN IF NOT EXISTS`` migration still
        # hydrate cleanly — they just see a ``None`` reason (same as a
        # never-set column on the current schema).
        return RepairRequest(
            id=str(row["id"]),
            chat_id=row["chat_id"],
            user_id=row["user_id"],
            mode=RequestMode(row["mode"]),
            status=RequestStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            timeout_at=row["timeout_at"],
            original_image_key=row["original_image_key"],
            composited_image_key=row["composited_image_key"],
            ml_error_code=_row_get(row, "ml_error_code"),
            ml_error_message=_row_get(row, "ml_error_message"),
            idempotency_key=row["idempotency_key"],
        )


def _row_get(row: asyncpg.Record, key: str) -> str | None:
    """Safe column accessor used by ``_from_row``.

    ``asyncpg.Record`` doesn't expose ``.get``; a missing key raises
    ``KeyError``. We use this to stay tolerant of pre-migration rows
    while the ALTER TABLE is still rolling out.
    """
    try:
        return row[key]
    except (KeyError, IndexError):
        return None
