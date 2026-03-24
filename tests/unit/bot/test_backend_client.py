"""Behavioral tests for BackendClient HTTP calls.

Uses httpx MockTransport to verify that the client sends correct requests
to the backend API.
"""

from __future__ import annotations

import json

import httpx
import pytest

from auto_repair_estimator.bot.backend_client import BackendClient


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _make_client(handler) -> BackendClient:
    client = BackendClient("http://test-backend:8000")
    client._client = httpx.AsyncClient(transport=_mock_transport(handler), base_url="http://test-backend:8000")
    return client


class TestCreateRequest:
    async def test_sends_correct_json_body(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"id": "req-1", "status": "created"})

        client = _make_client(handler)
        result = await client.create_request(chat_id=100, user_id=42, mode="ml")

        assert captured["body"] == {"chat_id": 100, "user_id": 42, "mode": "ml"}
        assert "/v1/requests" in captured["url"]
        assert result["id"] == "req-1"

    async def test_raises_on_server_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "internal"})

        client = _make_client(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await client.create_request(chat_id=1, user_id=None, mode="manual")


class TestUploadPhoto:
    async def test_sends_image_key_in_body(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "req-1", "status": "queued"})

        client = _make_client(handler)
        await client.upload_photo("req-1", "raw-images/req-1.jpg")

        assert captured["body"]["image_key"] == "raw-images/req-1.jpg"


class TestAddDamage:
    async def test_sends_part_and_damage_type(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "d1"})

        client = _make_client(handler)
        await client.add_damage("req-1", "hood", "scratch")

        assert captured["body"] == {"part_type": "hood", "damage_type": "scratch"}


class TestEditDamage:
    async def test_sends_patch_with_new_damage_type(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "d1"})

        client = _make_client(handler)
        await client.edit_damage("req-1", "d1", "dent")

        assert captured["method"] == "PATCH"
        assert captured["body"]["damage_type"] == "dent"


class TestDeleteDamage:
    async def test_sends_delete_request(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            return httpx.Response(204)

        client = _make_client(handler)
        await client.delete_damage("req-1", "d1")

        assert captured["method"] == "DELETE"
        assert "d1" in captured["url"]


class TestConfirmPricing:
    async def test_returns_pricing_result(self):
        expected = {"total_cost": 5000.0, "total_hours": 3.5, "breakdown": []}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=expected)

        client = _make_client(handler)
        result = await client.confirm_pricing("req-1")

        assert result["total_cost"] == 5000.0
        assert result["total_hours"] == 3.5
