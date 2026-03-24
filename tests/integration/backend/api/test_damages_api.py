"""Smoke-level API integration tests using in-memory repositories.

These tests do NOT require a running database. They validate that the HTTP
routing, request/response schema, and use-case wiring work correctly.
For full controller+database tests see ``test_api_with_database.py``.
"""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient
from pytest import mark

from auto_repair_estimator.backend.main import create_app


def _app_client() -> AsyncClient:
    """Return a TestClient backed by in-memory repositories."""
    app = create_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _create_manual_request(client: AsyncClient) -> str:
    resp = await client.post("/v1/requests", json={"chat_id": 1, "mode": "manual"})
    assert resp.status_code == 200
    return resp.json()["id"]


@mark.anyio
async def test_add_damage_to_manual_request_returns_correct_schema() -> None:
    # Arrange
    async with _app_client() as client:
        request_id = await _create_manual_request(client)

        # Act
        resp = await client.post(
            f"/v1/requests/{request_id}/damages",
            json={"part_type": "hood", "damage_type": "scratch"},
        )

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert data["part_type"] == "hood"
        assert data["damage_type"] == "scratch"
        assert data["source"] == "manual"
        assert not data["is_deleted"]


@mark.anyio
async def test_add_damage_to_nonexistent_request_returns_400() -> None:
    # Arrange
    async with _app_client() as client:
        # Act
        resp = await client.post(
            "/v1/requests/nonexistent/damages",
            json={"part_type": "hood", "damage_type": "scratch"},
        )

        # Assert
        assert resp.status_code == 400


@mark.anyio
async def test_edit_damage_type_returns_updated_resource() -> None:
    # Arrange
    async with _app_client() as client:
        request_id = await _create_manual_request(client)
        add_resp = await client.post(
            f"/v1/requests/{request_id}/damages",
            json={"part_type": "hood", "damage_type": "scratch"},
        )
        damage_id = add_resp.json()["id"]

        # Act
        edit_resp = await client.patch(
            f"/v1/requests/{request_id}/damages/{damage_id}",
            json={"damage_type": "dent"},
        )

        # Assert
        assert edit_resp.status_code == 200
        assert edit_resp.json()["damage_type"] == "dent"


@mark.anyio
async def test_delete_damage_returns_204_no_body() -> None:
    # Arrange
    async with _app_client() as client:
        request_id = await _create_manual_request(client)
        add_resp = await client.post(
            f"/v1/requests/{request_id}/damages",
            json={"part_type": "hood", "damage_type": "scratch"},
        )
        damage_id = add_resp.json()["id"]

        # Act
        del_resp = await client.delete(f"/v1/requests/{request_id}/damages/{damage_id}")

        # Assert
        assert del_resp.status_code == 204
        assert del_resp.content == b""


@mark.anyio
async def test_get_request_includes_added_damages_in_response() -> None:
    # Arrange
    async with _app_client() as client:
        request_id = await _create_manual_request(client)
        await client.post(
            f"/v1/requests/{request_id}/damages",
            json={"part_type": "bumper_front", "damage_type": "dent"},
        )

        # Act
        get_resp = await client.get(f"/v1/requests/{request_id}")

        # Assert
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["id"] == request_id
        assert len(data["damages"]) == 1
        assert data["damages"][0]["part_type"] == "bumper_front"


@mark.anyio
async def test_confirm_pricing_with_damage_returns_done_status() -> None:
    # Arrange
    async with _app_client() as client:
        request_id = await _create_manual_request(client)
        await client.post(
            f"/v1/requests/{request_id}/damages",
            json={"part_type": "hood", "damage_type": "scratch"},
        )

        # Act
        confirm_resp = await client.post(f"/v1/requests/{request_id}/confirm")

        # Assert
        assert confirm_resp.status_code == 200
        data = confirm_resp.json()
        assert data["status"] == "done"
        assert "total_cost" in data
        assert "total_hours" in data


@mark.anyio
async def test_confirm_pricing_for_nonexistent_request_returns_400() -> None:
    # Arrange
    async with _app_client() as client:
        # Act
        resp = await client.post("/v1/requests/nonexistent/confirm")

        # Assert
        assert resp.status_code == 400
