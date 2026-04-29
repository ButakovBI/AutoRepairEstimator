"""Adversarial tests for ``bot.handlers.photo.handle_photo``.

The handler operates on user-supplied data through several third parties
(VK photo CDN, httpx, backend HTTP API, MinIO via presigned URL). Each
step can fail independently; a failure on any of them must produce a
single clear "не удалось" reply rather than a silent drop or a crashed
coroutine.

Since the single-active-session refactor the handler processes **one
photo per message** even when the VK payload carries several — so
these tests focus on that single-photo contract under adversarial
conditions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.handlers.photo import handle_photo


class _FlakyHttpxClientFactory:
    """Factory returning scripted ``_FakeAsyncClient`` instances.

    Each invocation of ``httpx.AsyncClient()`` pops the next script —
    ``handle_photo`` opens one client for the VK GET and a second one
    for the presigned PUT, so tests encode both.
    """

    def __init__(self, scripts: list[dict]) -> None:
        self._scripts = scripts
        self._call_index = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        if self._call_index >= len(self._scripts):
            raise AssertionError(
                f"More httpx.AsyncClient() calls than scripted: idx={self._call_index}"
            )
        script = self._scripts[self._call_index]
        self._call_index += 1
        return _FakeAsyncClient(script)


class _FakeAsyncClient:
    def __init__(self, script: dict) -> None:
        self._script = script

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def get(self, url: str) -> MagicMock:
        if self._script.get("get_raises"):
            raise self._script["get_raises"]
        resp = MagicMock()
        resp.content = self._script.get("get_content", b"fake-jpeg-bytes")
        resp.raise_for_status = MagicMock()
        if self._script.get("get_http_error"):
            resp.raise_for_status.side_effect = self._script["get_http_error"]
        return resp

    async def put(self, url: str, **kwargs: object) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if self._script.get("put_http_error"):
            resp.raise_for_status.side_effect = self._script["put_http_error"]
        return resp


def _photo(url: str, width: int = 800, height: int = 600, sizes: list | None = None) -> MagicMock:
    photo = MagicMock()
    if sizes is None:
        size = MagicMock()
        size.url = url
        size.width = width
        size.height = height
        photo.sizes = [size]
    else:
        photo.sizes = sizes
    return photo


def _message(photos: list) -> MagicMock:
    msg = MagicMock()
    msg.peer_id = 10
    msg.from_id = 20
    msg.conversation_message_id = 1001
    msg.get_photo_attachments = MagicMock(return_value=photos)
    msg.answer = AsyncMock()
    return msg


def _backend() -> AsyncMock:
    backend = AsyncMock(spec=BackendClient)
    # No prior active session by default — the abandon-before-create
    # path is exercised in its own dedicated suite.
    backend.get_active_request = AsyncMock(return_value=None)
    backend.abandon_request = AsyncMock(return_value={"id": "x", "status": "failed"})

    async def create_request(**_: object) -> dict:
        return {"id": "req-1", "presigned_put_url": "https://s3/put/1"}

    backend.create_request = AsyncMock(side_effect=create_request)
    backend.upload_photo = AsyncMock()
    return backend


@pytest.mark.anyio
async def test_vk_download_failure_produces_user_visible_error() -> None:
    # GET from VK CDN times out → upload_photo must not fire (nothing
    # to upload) and the user sees a clean "не удалось" reply rather
    # than a silent drop.
    message = _message([_photo("https://vk.test/boom.jpg")])
    backend = _backend()

    scripts = [{"get_raises": httpx.ConnectTimeout("boom")}]

    with patch(
        "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
        _FlakyHttpxClientFactory(scripts),
    ):
        await handle_photo(message, backend, MagicMock())

    backend.upload_photo.assert_not_awaited()
    sent_texts = " ".join(
        call.args[0].lower() for call in message.answer.await_args_list if call.args
    )
    assert "не удалось" in sent_texts


@pytest.mark.anyio
async def test_presigned_put_failure_produces_user_visible_error() -> None:
    # PUT to the presigned MinIO URL returns 500 → upload_photo must
    # not be called (otherwise the backend would enqueue an inference
    # for a missing image key and the request would timeout silently).
    message = _message([_photo("https://vk.test/good.jpg")])
    backend = _backend()

    scripts = [
        {},  # VK GET succeeds
        {
            "put_http_error": httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
        },
    ]

    with patch(
        "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
        _FlakyHttpxClientFactory(scripts),
    ):
        await handle_photo(message, backend, MagicMock())

    backend.upload_photo.assert_not_awaited()
    sent_texts = " ".join(
        call.args[0].lower() for call in message.answer.await_args_list if call.args
    )
    assert "не удалось" in sent_texts


@pytest.mark.anyio
async def test_photo_with_no_sizes_is_handled_gracefully() -> None:
    # A VK photo attachment can come back without any ``sizes`` entries
    # (moderation-removed, deleted). The handler raises before any
    # HTTP work and the user gets the generic failure reply.
    message = _message([_photo("https://vk.test/no-sizes.jpg", sizes=[])])
    backend = _backend()

    with patch(
        "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
        _FlakyHttpxClientFactory([]),
    ):
        await handle_photo(message, backend, MagicMock())

    backend.upload_photo.assert_not_awaited()
    sent_texts = " ".join(
        call.args[0].lower() for call in message.answer.await_args_list if call.args
    )
    assert "не удалось" in sent_texts


@pytest.mark.anyio
async def test_first_photo_failure_does_not_fall_through_to_second() -> None:
    # After the single-photo refactor we must NOT silently fall through
    # to photo #2 when photo #1 fails — that would reintroduce the
    # "parallel-sessions per message" bug in the error path.
    message = _message(
        [
            _photo("https://vk.test/broken.jpg"),
            _photo("https://vk.test/good.jpg"),
        ]
    )
    backend = _backend()

    scripts = [{"get_raises": httpx.ConnectTimeout("boom")}]

    with patch(
        "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
        _FlakyHttpxClientFactory(scripts),
    ):
        await handle_photo(message, backend, MagicMock())

    # Exactly one create_request (for photo #1), zero uploads.
    assert backend.create_request.await_count == 1
    backend.upload_photo.assert_not_awaited()
