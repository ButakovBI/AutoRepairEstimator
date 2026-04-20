"""Integration tests for ``GET /v1/requests/active``.

This endpoint underpins the bot's "there's no active session" detection.
The contract boils down to four observable behaviors:

1. No requests exist for the chat_id -> 404.
2. A non-terminal request exists for the chat_id -> 200 with its summary.
3. Only terminal requests (done/failed) exist -> 404.
4. Multiple non-terminal requests -> the newest is returned.
"""

from __future__ import annotations

import datetime as _dt
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from pytest import mark

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus
from auto_repair_estimator.backend.main import create_app


async def _seed_request(
    app,  # type: ignore[no-untyped-def]
    *,
    chat_id: int,
    status: RequestStatus,
    created_at: _dt.datetime | None = None,
) -> RepairRequest:
    now = created_at or _dt.datetime.now(_dt.UTC)
    request = RepairRequest(
        id=str(uuid4()),
        chat_id=chat_id,
        user_id=1,
        mode=RequestMode.MANUAL,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=now + _dt.timedelta(minutes=5),
    )
    await app.state.request_repo.add(request)
    return request


@mark.anyio
async def test_returns_404_when_no_requests_exist() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/requests/active", params={"chat_id": 9999})
    assert resp.status_code == 404


@mark.anyio
async def test_returns_active_non_terminal_request() -> None:
    app = create_app()
    seeded = await _seed_request(app, chat_id=77, status=RequestStatus.PRICING)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/requests/active", params={"chat_id": 77})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == seeded.id
    assert body["status"] == "pricing"


@mark.anyio
async def test_treats_done_and_failed_as_inactive() -> None:
    # Only terminal sessions exist for this chat -> no active session.
    app = create_app()
    await _seed_request(app, chat_id=55, status=RequestStatus.DONE)
    await _seed_request(app, chat_id=55, status=RequestStatus.FAILED)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/requests/active", params={"chat_id": 55})
    assert resp.status_code == 404


@mark.anyio
async def test_returns_most_recent_non_terminal_when_multiple_active() -> None:
    # If a user created two overlapping sessions, the newer one wins --
    # semantically "whatever they're doing *now*".
    app = create_app()
    old = _dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=10)
    newer = _dt.datetime.now(_dt.UTC)
    await _seed_request(app, chat_id=42, status=RequestStatus.PRICING, created_at=old)
    latest = await _seed_request(app, chat_id=42, status=RequestStatus.PRICING, created_at=newer)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/requests/active", params={"chat_id": 42})
    assert resp.status_code == 200
    assert resp.json()["id"] == latest.id
