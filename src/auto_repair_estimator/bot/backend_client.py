from __future__ import annotations

from typing import Any

import httpx


class BackendClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)

    async def create_request(self, chat_id: int, user_id: int | None, mode: str) -> dict[str, Any]:
        resp = await self._client.post(
            "/v1/requests",
            json={"chat_id": chat_id, "user_id": user_id, "mode": mode},
        )
        resp.raise_for_status()
        return resp.json()

    async def upload_photo(self, request_id: str, image_key: str) -> dict[str, Any]:
        resp = await self._client.post(
            f"/v1/requests/{request_id}/photo",
            json={"image_key": image_key},
        )
        resp.raise_for_status()
        return resp.json()

    async def get_request(self, request_id: str) -> dict[str, Any]:
        resp = await self._client.get(f"/v1/requests/{request_id}")
        resp.raise_for_status()
        return resp.json()

    async def add_damage(self, request_id: str, part_type: str, damage_type: str) -> dict[str, Any]:
        resp = await self._client.post(
            f"/v1/requests/{request_id}/damages",
            json={"part_type": part_type, "damage_type": damage_type},
        )
        resp.raise_for_status()
        return resp.json()

    async def edit_damage(self, request_id: str, damage_id: str, damage_type: str) -> dict[str, Any]:
        resp = await self._client.patch(
            f"/v1/requests/{request_id}/damages/{damage_id}",
            json={"damage_type": damage_type},
        )
        resp.raise_for_status()
        return resp.json()

    async def delete_damage(self, request_id: str, damage_id: str) -> None:
        resp = await self._client.delete(f"/v1/requests/{request_id}/damages/{damage_id}")
        resp.raise_for_status()

    async def confirm_pricing(self, request_id: str) -> dict[str, Any]:
        resp = await self._client.post(f"/v1/requests/{request_id}/confirm")
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
