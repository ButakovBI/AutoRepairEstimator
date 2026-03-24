"""Controller + database integration tests.

These tests verify the behaviour visible through the HTTP interface while using a
real PostgreSQL database (provided by the ``api_client`` fixture from conftest.py).

Focus: each test covers one user-facing behaviour end-to-end through the
controller layer and asserts on the final database state where meaningful.
All tests follow the AAA (Arrange / Act / Assert) convention.
"""

from __future__ import annotations

import asyncpg
import pytest
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_manual_request(client: AsyncClient) -> str:
    resp = await client.post("/v1/requests", json={"chat_id": 1, "mode": "manual"})
    assert resp.status_code == 200
    return resp.json()["id"]


async def _create_ml_request(client: AsyncClient) -> str:
    resp = await client.post("/v1/requests", json={"chat_id": 2, "user_id": 99, "mode": "ml"})
    assert resp.status_code == 200
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_manual_request_is_created_with_pricing_status(api_client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    # Arrange — nothing to arrange, client is fresh

    # Act
    resp = await api_client.post("/v1/requests", json={"chat_id": 5, "mode": "manual"})

    # Assert — HTTP layer
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "manual"
    assert body["status"] == "pricing"

    # Assert — database reflects same state
    row = await db_pool.fetchrow("SELECT status FROM repair_requests WHERE id = $1", body["id"])
    assert row is not None
    assert row["status"] == "pricing"


@pytest.mark.anyio
async def test_ml_request_is_created_with_created_status(api_client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    # Arrange / Act
    resp = await api_client.post("/v1/requests", json={"chat_id": 6, "user_id": 7, "mode": "ml"})

    # Assert — HTTP
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "created"

    # Assert — database
    row = await db_pool.fetchrow("SELECT status, mode FROM repair_requests WHERE id = $1", body["id"])
    assert row is not None
    assert row["status"] == "created"
    assert row["mode"] == "ml"


@pytest.mark.anyio
async def test_photo_upload_transitions_ml_request_to_queued(api_client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    # Arrange
    request_id = await _create_ml_request(api_client)

    # Act
    resp = await api_client.post(f"/v1/requests/{request_id}/photo", json={"image_key": "raw-images/test.jpg"})

    # Assert — HTTP
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"

    # Assert — database has the image key stored
    row = await db_pool.fetchrow("SELECT status, original_image_key FROM repair_requests WHERE id = $1", request_id)
    assert row is not None
    assert row["status"] == "queued"
    assert row["original_image_key"] == "raw-images/test.jpg"


@pytest.mark.anyio
async def test_photo_upload_rejected_for_manual_mode(api_client: AsyncClient) -> None:
    # Arrange
    request_id = await _create_manual_request(api_client)

    # Act — try to upload a photo to a manual request
    resp = await api_client.post(f"/v1/requests/{request_id}/photo", json={"image_key": "raw-images/bad.jpg"})

    # Assert — controller must reject with 400
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_confirm_pricing_transitions_request_to_done(api_client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    # Arrange
    request_id = await _create_manual_request(api_client)

    # Act
    resp = await api_client.post(f"/v1/requests/{request_id}/confirm")

    # Assert — HTTP
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert "total_cost" in body
    assert "total_hours" in body

    # Assert — database
    row = await db_pool.fetchrow("SELECT status FROM repair_requests WHERE id = $1", request_id)
    assert row is not None
    assert row["status"] == "done"


# ---------------------------------------------------------------------------
# Damage management
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_add_damage_persists_record_in_database(api_client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    # Arrange
    request_id = await _create_manual_request(api_client)

    # Act
    resp = await api_client.post(
        f"/v1/requests/{request_id}/damages", json={"part_type": "hood", "damage_type": "scratch"}
    )

    # Assert — HTTP
    assert resp.status_code == 200
    damage_id = resp.json()["id"]

    # Assert — row exists in database
    row = await db_pool.fetchrow(
        "SELECT damage_type, part_type, is_deleted FROM detected_damages WHERE id = $1", damage_id
    )
    assert row is not None
    assert row["damage_type"] == "scratch"
    assert row["part_type"] == "hood"
    assert row["is_deleted"] is False


@pytest.mark.anyio
async def test_edit_damage_type_updates_database_record(api_client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    # Arrange
    request_id = await _create_manual_request(api_client)
    add_resp = await api_client.post(
        f"/v1/requests/{request_id}/damages", json={"part_type": "hood", "damage_type": "scratch"}
    )
    damage_id = add_resp.json()["id"]

    # Act
    edit_resp = await api_client.patch(f"/v1/requests/{request_id}/damages/{damage_id}", json={"damage_type": "dent"})

    # Assert — HTTP
    assert edit_resp.status_code == 200
    assert edit_resp.json()["damage_type"] == "dent"

    # Assert — database reflects the update
    row = await db_pool.fetchrow("SELECT damage_type FROM detected_damages WHERE id = $1", damage_id)
    assert row is not None
    assert row["damage_type"] == "dent"


@pytest.mark.anyio
async def test_delete_damage_soft_deletes_in_database(api_client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    # Arrange
    request_id = await _create_manual_request(api_client)
    add_resp = await api_client.post(
        f"/v1/requests/{request_id}/damages", json={"part_type": "trunk", "damage_type": "rust"}
    )
    damage_id = add_resp.json()["id"]

    # Act
    del_resp = await api_client.delete(f"/v1/requests/{request_id}/damages/{damage_id}")

    # Assert — HTTP returns 204 with no body
    assert del_resp.status_code == 204

    # Assert — record still exists in DB but is flagged deleted (soft delete)
    row = await db_pool.fetchrow("SELECT is_deleted FROM detected_damages WHERE id = $1", damage_id)
    assert row is not None
    assert row["is_deleted"] is True


@pytest.mark.anyio
async def test_get_request_returns_all_damages_with_correct_data(
    api_client: AsyncClient, db_pool: asyncpg.Pool
) -> None:
    # Arrange
    request_id = await _create_manual_request(api_client)
    await api_client.post(
        f"/v1/requests/{request_id}/damages", json={"part_type": "bumper_front", "damage_type": "dent"}
    )
    await api_client.post(f"/v1/requests/{request_id}/damages", json={"part_type": "hood", "damage_type": "scratch"})

    # Act
    get_resp = await api_client.get(f"/v1/requests/{request_id}")

    # Assert
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["id"] == request_id
    assert len(data["damages"]) == 2
    part_types = {d["part_type"] for d in data["damages"]}
    assert part_types == {"bumper_front", "hood"}


@pytest.mark.anyio
async def test_confirm_pricing_with_damages_uses_pricing_rules(api_client: AsyncClient, db_pool: asyncpg.Pool) -> None:
    """Pricing is calculated from real pricing_rules table in the database."""
    # Arrange — a manual request with one known damage
    request_id = await _create_manual_request(api_client)
    # hood + scratch = 1.5 h, 1200 RUB (see init.sql)
    await api_client.post(f"/v1/requests/{request_id}/damages", json={"part_type": "hood", "damage_type": "scratch"})

    # Act
    resp = await api_client.post(f"/v1/requests/{request_id}/confirm")

    # Assert — pricing reflects the rule from the seeded database
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_hours"] == pytest.approx(1.5)
    assert body["total_cost"] == pytest.approx(1200.0)
    assert body["status"] == "done"


@pytest.mark.anyio
async def test_add_damage_to_nonexistent_request_returns_400(api_client: AsyncClient) -> None:
    # Arrange — no request created

    # Act
    resp = await api_client.post(
        "/v1/requests/00000000-0000-0000-0000-000000000000/damages",
        json={"part_type": "hood", "damage_type": "scratch"},
    )

    # Assert
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_get_nonexistent_request_returns_404(api_client: AsyncClient) -> None:
    # Act
    resp = await api_client.get("/v1/requests/00000000-0000-0000-0000-000000000000")

    # Assert
    assert resp.status_code == 404
