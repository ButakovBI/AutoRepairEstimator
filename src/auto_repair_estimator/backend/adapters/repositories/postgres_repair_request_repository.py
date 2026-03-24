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
                 original_image_key, composited_image_key, idempotency_key)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
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
            None,
        )

    async def get(self, request_id: str) -> RepairRequest | None:
        row = await self._pool.fetchrow("SELECT * FROM repair_requests WHERE id = $1", request_id)
        if row is None:
            return None
        return self._from_row(row)

    async def update(self, request: RepairRequest) -> None:
        logger.debug("UPDATE repair_request id={} status={}", request.id, request.status)
        await self._pool.execute(
            """
            UPDATE repair_requests
            SET status=$2, updated_at=$3, original_image_key=$4,
                composited_image_key=$5, timeout_at=$6
            WHERE id=$1
            """,
            request.id,
            request.status.value,
            request.updated_at,
            request.original_image_key,
            request.composited_image_key,
            request.timeout_at,
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

    @staticmethod
    def _from_row(row: asyncpg.Record) -> RepairRequest:
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
            ml_error_code=None,
            ml_error_message=None,
        )
