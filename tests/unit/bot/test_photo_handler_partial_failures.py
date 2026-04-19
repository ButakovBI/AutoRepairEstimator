"""Adversarial tests for ``bot.handlers.photo.handle_photo``.

The handler operates on user-supplied data through several third parties
(VK photo CDN, httpx, backend HTTP API, MinIO via presigned URL). Each
step can fail independently; the handler must keep processing remaining
photos and report accurate success counts to the user, without ever
leaving the conversation silent.

The tests below verify four specific adversarial scenarios that are not
covered by the happy-path suite in ``test_handlers.py``:

1. One photo download fails mid-loop — other photos still processed, user
   sees a summary with the correct success count.
2. The presigned PUT to MinIO returns 500 — that photo fails but the
   remaining photos upload successfully.
3. The VK photo has an empty ``sizes`` list (can happen for deleted photos
   or moderation-blocked content) — the handler skips it gracefully.
4. All photos fail — the user receives a clear error message instead of
   a success summary.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.handlers.photo import handle_photo


class _FlakyHttpxClientFactory:
    """Factory returning ``_FakeAsyncClient`` instances with configurable
    failure behaviour per invocation.

    Each test instantiates this factory with a list of scripted responses
    (one per httpx.AsyncClient context); the handler opens a new client
    per GET and a new one per PUT, so we encode that sequence explicitly.
    """

    def __init__(self, scripts: list[dict]) -> None:
        self._scripts = scripts
        self._call_index = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        if self._call_index >= len(self._scripts):
            raise AssertionError(
                f"Handler made more httpx.AsyncClient() calls than scripted: index={self._call_index}, "
                f"scripts={len(self._scripts)}"
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
    msg.get_photo_attachments = MagicMock(return_value=photos)
    msg.answer = AsyncMock()
    return msg


def _backend() -> AsyncMock:
    backend = AsyncMock(spec=BackendClient)
    # Every call returns a fresh id so the handler uploads to unique keys.
    counter = {"i": 0}

    async def create_request(**_: object) -> dict:
        counter["i"] += 1
        return {"id": f"req-{counter['i']}", "presigned_put_url": f"https://s3/put/{counter['i']}"}

    backend.create_request = AsyncMock(side_effect=create_request)
    backend.upload_photo = AsyncMock()
    return backend


@pytest.mark.anyio
async def test_single_failed_photo_does_not_block_the_others() -> None:
    """Photo 1 download fails, photos 2 and 3 succeed.

    Expectation: ``create_request``/``upload_photo`` each invoked twice
    (only for the successes), the user sees a summary message saying
    "2 из 3".
    """

    message = _message(
        [
            _photo("https://vk.test/broken.jpg"),
            _photo("https://vk.test/ok-2.jpg"),
            _photo("https://vk.test/ok-3.jpg"),
        ]
    )
    backend = _backend()

    scripts = [
        # Photo 1: GET raises ConnectTimeout.
        {"get_raises": httpx.ConnectTimeout("boom")},
        # Photo 2: GET ok, PUT ok.
        {},
        {},
        # Photo 3: GET ok, PUT ok.
        {},
        {},
    ]

    with patch(
        "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
        _FlakyHttpxClientFactory(scripts),
    ):
        await handle_photo(message, backend, MagicMock())

    assert backend.create_request.await_count == 3, (
        "Handler must call create_request once per attachment even if some downloads fail."
    )
    assert backend.upload_photo.await_count == 2, (
        "Only successfully-uploaded photos must confirm to the backend. "
        f"Got {backend.upload_photo.await_count}."
    )

    sent_texts = [call.args[0] for call in message.answer.await_args_list if call.args]
    summary = sent_texts[-1].lower()
    assert "2 из 3" in summary or "2 из 3".replace(" ", "") in summary.replace(" ", "")


@pytest.mark.anyio
async def test_presigned_put_failure_on_one_photo_does_not_lose_the_other() -> None:
    message = _message(
        [
            _photo("https://vk.test/good.jpg"),
            _photo("https://vk.test/put-broken.jpg"),
        ]
    )
    backend = _backend()

    scripts = [
        # Photo 1 GET, PUT — both succeed.
        {},
        {},
        # Photo 2 GET — succeeds.
        {},
        # Photo 2 PUT — returns 500.
        {"put_http_error": httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())},
    ]

    with patch(
        "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
        _FlakyHttpxClientFactory(scripts),
    ):
        await handle_photo(message, backend, MagicMock())

    # Only the successful photo's upload was confirmed.
    assert backend.upload_photo.await_count == 1
    # Both create_request calls still happened (backend state is then
    # timed out by HeartbeatChecker; that's acceptable).
    assert backend.create_request.await_count == 2


@pytest.mark.anyio
async def test_photo_with_no_sizes_is_skipped_gracefully() -> None:
    """A VK photo attachment can come back without any ``sizes`` entries
    (moderation-removed, deleted). The handler must not explode.
    """

    message = _message(
        [
            _photo("https://vk.test/no-sizes.jpg", sizes=[]),
            _photo("https://vk.test/good.jpg"),
        ]
    )
    backend = _backend()

    # Only one successful download/upload cycle (for the good photo). The
    # no-sizes photo fails at the ``sizes`` check before any HTTP call,
    # so the http factory sees only 2 scripts (GET, PUT for photo 2).
    scripts = [{}, {}]

    with patch(
        "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
        _FlakyHttpxClientFactory(scripts),
    ):
        await handle_photo(message, backend, MagicMock())

    assert backend.upload_photo.await_count == 1


@pytest.mark.anyio
async def test_all_photos_failing_produces_user_visible_error_message() -> None:
    message = _message([_photo("https://vk.test/a.jpg"), _photo("https://vk.test/b.jpg")])
    backend = _backend()

    scripts = [
        {"get_raises": httpx.ConnectTimeout("boom")},
        {"get_raises": httpx.ConnectTimeout("boom")},
    ]

    with patch(
        "auto_repair_estimator.bot.handlers.photo.httpx.AsyncClient",
        _FlakyHttpxClientFactory(scripts),
    ):
        await handle_photo(message, backend, MagicMock())

    # No upload confirmations.
    assert backend.upload_photo.await_count == 0
    # User received a clear failure message (Russian keyword "не удалось").
    sent_texts = " ".join(
        call.args[0].lower() for call in message.answer.await_args_list if call.args
    )
    assert "не удалось" in sent_texts, (
        f"When every photo fails the user must see an error, not silence. Sent: {sent_texts!r}"
    )
