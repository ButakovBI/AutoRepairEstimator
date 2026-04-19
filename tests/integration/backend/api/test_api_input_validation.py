"""Adversarial HTTP-level tests pinning the FastAPI validation layer.

These tests run against the in-memory app (no database, no storage) so
they're fast and stable. Their job is to prove that every user-exposed
field is **validated at the edge**, not in the use cases — a sloppy
schema would let garbage reach the repositories where a ValueError would
become a 500 and leak internals.
"""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient
from pytest import mark

from auto_repair_estimator.backend.main import create_app


async def _client() -> AsyncClient:
    app = create_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@mark.anyio
async def test_create_request_missing_chat_id_returns_422() -> None:
    async with await _client() as client:
        resp = await client.post("/v1/requests", json={"user_id": 2, "mode": "ml"})

    assert resp.status_code == 422
    body = resp.json()
    # FastAPI surfaces the missing field in the error detail array.
    fields = {tuple(err["loc"])[-1] for err in body["detail"]}
    assert "chat_id" in fields


@mark.anyio
async def test_create_request_invalid_mode_returns_422() -> None:
    async with await _client() as client:
        # ``mode`` must be one of the RequestMode enum values.
        resp = await client.post(
            "/v1/requests",
            json={"chat_id": 1, "user_id": 2, "mode": "telepathic_mode"},
        )

    assert resp.status_code == 422


@mark.anyio
async def test_add_damage_invalid_part_type_returns_422() -> None:
    async with await _client() as client:
        # First create a manual request so the id is valid — otherwise we'd
        # get 400 from the use case and not exercise the validation path.
        create = await client.post(
            "/v1/requests", json={"chat_id": 1, "user_id": 2, "mode": "manual"}
        )
        rid = create.json()["id"]

        # ``windshield_wiper`` is not in ``PartType`` (user-approved enum).
        resp = await client.post(
            f"/v1/requests/{rid}/damages",
            json={"part_type": "windshield_wiper", "damage_type": "scratch"},
        )

    assert resp.status_code == 422


@mark.anyio
async def test_add_damage_invalid_damage_type_returns_422() -> None:
    async with await _client() as client:
        create = await client.post(
            "/v1/requests", json={"chat_id": 1, "user_id": 2, "mode": "manual"}
        )
        rid = create.json()["id"]

        resp = await client.post(
            f"/v1/requests/{rid}/damages",
            # ``tear`` was intentionally excluded from the damage enum per
            # the user-authoritative list — this protects that decision.
            json={"part_type": "hood", "damage_type": "tear"},
        )

    assert resp.status_code == 422


@mark.anyio
async def test_upload_photo_missing_image_key_returns_422() -> None:
    async with await _client() as client:
        create = await client.post(
            "/v1/requests", json={"chat_id": 1, "user_id": 2, "mode": "ml"}
        )
        rid = create.json()["id"]

        resp = await client.post(f"/v1/requests/{rid}/photo", json={})

    assert resp.status_code == 422


@mark.anyio
async def test_get_nonexistent_request_returns_404_not_500() -> None:
    async with await _client() as client:
        resp = await client.get("/v1/requests/definitely-not-a-real-id")

    assert resp.status_code == 404


@mark.anyio
async def test_upload_photo_for_nonexistent_request_returns_400_not_500() -> None:
    async with await _client() as client:
        resp = await client.post(
            "/v1/requests/nope/photo",
            json={"image_key": "raw-images/whatever.jpg"},
        )

    assert resp.status_code == 400, (
        "Unknown request id must be a client error (400), not a server crash. "
        f"Got {resp.status_code}: {resp.text}"
    )


@mark.anyio
async def test_edit_damage_rejects_unknown_damage_type_with_422() -> None:
    """Regression guard: the PATCH endpoint must validate the new enum."""
    async with await _client() as client:
        create = await client.post(
            "/v1/requests", json={"chat_id": 1, "user_id": 2, "mode": "manual"}
        )
        rid = create.json()["id"]
        # Add a damage so we have a valid damage id to patch against.
        add = await client.post(
            f"/v1/requests/{rid}/damages",
            json={"part_type": "hood", "damage_type": "scratch"},
        )
        did = add.json()["id"]

        resp = await client.patch(
            f"/v1/requests/{rid}/damages/{did}",
            json={"damage_type": "poltergeist"},
        )

    assert resp.status_code == 422


@mark.anyio
async def test_edit_damage_accepts_optional_part_type_for_reassignment() -> None:
    """Spec §3: "Изменить принадлежность повреждения к детали".

    The PATCH endpoint must accept a new ``part_type`` and persist it.
    This is the HTTP-layer counterpart of
    ``test_edit_damage_part_reassignment.py`` — makes sure the wire
    contract wasn't reverted.
    """

    async with await _client() as client:
        create = await client.post(
            "/v1/requests", json={"chat_id": 1, "user_id": 2, "mode": "manual"}
        )
        rid = create.json()["id"]
        add = await client.post(
            f"/v1/requests/{rid}/damages",
            json={"part_type": "hood", "damage_type": "scratch"},
        )
        did = add.json()["id"]

        resp = await client.patch(
            f"/v1/requests/{rid}/damages/{did}",
            json={"damage_type": "dent", "part_type": "bumper"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["damage_type"] == "dent"
    assert body["part_type"] == "bumper"
