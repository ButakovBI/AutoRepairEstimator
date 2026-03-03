from httpx import AsyncClient
from pytest import mark

import auto_repair_estimator.backend.api as requests_api
from auto_repair_estimator.backend.main import create_app
from auto_repair_estimator.backend.adapters.repositories.in_memory_repair_request_repository import (
    InMemoryRepairRequestRepository,
)


def _create_app():
    requests_api._repository = InMemoryRepairRequestRepository()
    return create_app()


@mark.anyio
async def test_create_ml_request_and_upload_photo_flow() -> None:
    backend_app = _create_app()
    async with AsyncClient(app=backend_app, base_url="http://test") as client:
        create_response = await client.post(
            "/v1/requests",
            json={"telegram_chat_id": 1, "telegram_user_id": 2, "mode": "ml"},
        )
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["mode"] == "ml"
        assert created["status"] == "created"

        request_id = created["id"]
        upload_response = await client.post(
            f"/v1/requests/{request_id}/photo",
            json={"image_key": "raw-images/request-1.jpg"},
        )
        assert upload_response.status_code == 200
        uploaded = upload_response.json()
        assert uploaded["id"] == request_id
        assert uploaded["status"] == "queued"


@mark.anyio
async def test_create_manual_request_and_confirm_pricing_flow() -> None:
    backend_app = _create_app()
    async with AsyncClient(app=backend_app, base_url="http://test") as client:
        create_response = await client.post(
            "/v1/requests",
            json={"telegram_chat_id": 1, "telegram_user_id": None, "mode": "manual"},
        )
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["mode"] == "manual"
        assert created["status"] == "pricing"

        request_id = created["id"]
        pricing_response = await client.post(f"/v1/requests/{request_id}/confirm-pricing")
        assert pricing_response.status_code == 200
        payload = pricing_response.json()
        assert payload["id"] == request_id
        assert payload["status"] == "done"
        assert payload["total_cost"] == 0.0
        assert payload["total_hours"] == 0.0

