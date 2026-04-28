from __future__ import annotations

from typing import Any, cast

import httpx


class BackendClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)

    @staticmethod
    def _json_object(resp: httpx.Response) -> dict[str, Any]:
        return cast(dict[str, Any], resp.json())

    async def create_request(
        self,
        chat_id: int,
        user_id: int | None,
        mode: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"chat_id": chat_id, "user_id": user_id, "mode": mode}
        if idempotency_key is not None:
            body["idempotency_key"] = idempotency_key
        resp = await self._client.post("/v1/requests", json=body)
        resp.raise_for_status()
        return self._json_object(resp)

    async def upload_photo(self, request_id: str, image_key: str) -> dict[str, Any]:
        resp = await self._client.post(
            f"/v1/requests/{request_id}/photo",
            json={"image_key": image_key},
        )
        resp.raise_for_status()
        return self._json_object(resp)

    async def get_request(self, request_id: str) -> dict[str, Any]:
        resp = await self._client.get(f"/v1/requests/{request_id}")
        resp.raise_for_status()
        return self._json_object(resp)

    async def get_active_request(self, chat_id: int) -> dict[str, Any] | None:
        """Fetch the user's latest non-terminal session or ``None``.

        A 404 from the backend is part of the contract — it means the user
        has no active scenario — so we translate it to ``None`` rather than
        bubbling an HTTPStatusError, which would force every caller to
        duplicate the same ``try / except`` around a control-flow signal.
        """
        resp = await self._client.get("/v1/requests/active", params={"chat_id": chat_id})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return self._json_object(resp)

    async def add_damage(self, request_id: str, part_type: str, damage_type: str) -> dict[str, Any]:
        resp = await self._client.post(
            f"/v1/requests/{request_id}/damages",
            json={"part_type": part_type, "damage_type": damage_type},
        )
        resp.raise_for_status()
        return self._json_object(resp)

    async def edit_damage(
        self,
        request_id: str,
        damage_id: str,
        damage_type: str,
        part_type: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"damage_type": damage_type}
        if part_type is not None:
            payload["part_type"] = part_type
        resp = await self._client.patch(
            f"/v1/requests/{request_id}/damages/{damage_id}",
            json=payload,
        )
        resp.raise_for_status()
        return self._json_object(resp)

    async def delete_damage(self, request_id: str, damage_id: str) -> None:
        resp = await self._client.delete(f"/v1/requests/{request_id}/damages/{damage_id}")
        resp.raise_for_status()

    async def confirm_pricing(self, request_id: str) -> dict[str, Any]:
        resp = await self._client.post(f"/v1/requests/{request_id}/confirm")
        resp.raise_for_status()
        return self._json_object(resp)

    async def abandon_request(self, request_id: str) -> dict[str, Any]:
        """Explicitly mark a session as FAILED (``user_abandoned``).

        The bot calls this whenever the user presses "Начать" or switches
        modes while an older session is still non-terminal, so the two
        never coexist. The endpoint is idempotent server-side: calling on
        an already-terminal request returns ``was_already_terminal=True``
        with 200, so callers don't need to branch on the current status.
        """
        resp = await self._client.post(f"/v1/requests/{request_id}/abandon")
        resp.raise_for_status()
        return self._json_object(resp)

    async def aclose(self) -> None:
        await self._client.aclose()
