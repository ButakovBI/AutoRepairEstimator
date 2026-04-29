"""Smoke-level API integration tests for repair-request lifecycle.

These use in-memory repositories so no database is required.
Full controller+database coverage is in ``test_api_with_database.py``.
"""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient
from pytest import mark

from auto_repair_estimator.backend.main import create_app


@mark.anyio
async def test_create_ml_request_returns_created_status() -> None:
    # Arrange
    app = create_app()

    # Act
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/requests",
            json={"chat_id": 1, "user_id": 2, "mode": "ml"},
        )

    # Assert
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "ml"
    assert body["status"] == "created"


@mark.anyio
async def test_photo_upload_transitions_ml_request_to_queued() -> None:
    # Arrange
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create_resp = await client.post("/v1/requests", json={"chat_id": 1, "user_id": 2, "mode": "ml"})
        request_id = create_resp.json()["id"]

        # Act
        upload_resp = await client.post(
            f"/v1/requests/{request_id}/photo", json={"image_key": "raw-images/request-1.jpg"}
        )

    # Assert
    assert upload_resp.status_code == 200
    body = upload_resp.json()
    assert body["id"] == request_id
    assert body["status"] == "queued"


@mark.anyio
async def test_create_manual_request_returns_pricing_status() -> None:
    # Arrange
    app = create_app()

    # Act
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/requests", json={"chat_id": 1, "user_id": None, "mode": "manual"})

    # Assert
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "manual"
    assert body["status"] == "pricing"


@mark.anyio
async def test_confirm_pricing_transitions_manual_request_to_done() -> None:
    # Arrange
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create_resp = await client.post("/v1/requests", json={"chat_id": 1, "user_id": None, "mode": "manual"})
        request_id = create_resp.json()["id"]

        # Act
        pricing_resp = await client.post(f"/v1/requests/{request_id}/confirm-pricing")

    # Assert
    assert pricing_resp.status_code == 200
    body = pricing_resp.json()
    assert body["id"] == request_id
    assert body["status"] == "done"
    # No damages were registered, so every bound of every range is zero.
    assert body["total_cost_min"] == 0.0
    assert body["total_cost_max"] == 0.0
    assert body["total_hours_min"] == 0.0
    assert body["total_hours_max"] == 0.0
    assert body["breakdown"] == []
