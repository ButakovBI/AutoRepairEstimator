"""Adversarial tests for ``InferenceResultConsumer._handle``.

The consumer is the untrusted-input boundary from the ML worker. It must
refuse to crash on malformed Kafka messages because a single unhandled
exception would kill the whole consumer coroutine (see the ``async for`` in
``run``), stalling inference-result ingestion for every user.

These tests exercise ``_handle`` directly with dict payloads — the Kafka
transport details are the gateway's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from auto_repair_estimator.backend.use_cases.process_inference_result import (
    ProcessInferenceResultInput,
)
from auto_repair_estimator.backend.workers.inference_result_consumer import InferenceResultConsumer


@dataclass
class _RecordedCall:
    data: ProcessInferenceResultInput


class _StubUseCase:
    """Captures each ``execute`` call so the test can assert on the payload
    the use case received — this is the unit's real contract."""

    def __init__(self) -> None:
        self.calls: list[_RecordedCall] = []

    async def execute(self, data: ProcessInferenceResultInput) -> None:
        self.calls.append(_RecordedCall(data=data))


def _consumer(use_case: _StubUseCase) -> InferenceResultConsumer:
    # The ``KafkaConsumer`` parameter is irrelevant for ``_handle`` unit
    # tests; ``None`` keeps the fixture minimal.
    return InferenceResultConsumer(consumer=None, use_case=use_case)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_valid_success_message_is_dispatched_to_use_case() -> None:
    use_case = _StubUseCase()
    consumer = _consumer(use_case)
    message: dict[str, Any] = {
        "request_id": "req-1",
        "status": "success",
        "parts": [
            {"part_type": "hood", "confidence": 0.9, "bbox": [0.5, 0.5, 0.3, 0.3], "crop_image_key": "c1"}
        ],
        "damages": [
            {"damage_type": "scratch", "part_type": "hood", "confidence": 0.8, "mask_image_key": "m1"}
        ],
        "composited_image_key": "composite/req-1.jpg",
    }

    await consumer._handle(message)

    assert len(use_case.calls) == 1
    received = use_case.calls[0].data
    assert received.request_id == "req-1"
    assert received.status == "success"
    assert received.composited_image_key == "composite/req-1.jpg"
    assert [p.part_type for p in received.parts] == ["hood"]
    assert [d.damage_type for d in received.damages] == ["scratch"]


@pytest.mark.anyio
async def test_missing_request_id_is_skipped_without_invoking_use_case() -> None:
    use_case = _StubUseCase()
    consumer = _consumer(use_case)
    message: dict[str, Any] = {
        # No "request_id" key at all — older ML worker versions could send
        # this by mistake, or an attacker could replay a stripped payload.
        "status": "success",
        "parts": [],
        "damages": [],
    }

    await consumer._handle(message)

    assert use_case.calls == [], (
        "A message without request_id must be dropped before the use case "
        "runs, otherwise the use case would attempt to look up an empty id."
    )


@pytest.mark.anyio
async def test_missing_confidence_defaults_to_zero_instead_of_crashing() -> None:
    # ML worker may forget to populate ``confidence`` on some code paths.
    # The consumer must coerce to 0.0 rather than raising KeyError — the
    # use case still runs, the detection is kept.
    use_case = _StubUseCase()
    consumer = _consumer(use_case)
    message: dict[str, Any] = {
        "request_id": "req-2",
        "status": "success",
        "parts": [{"part_type": "hood"}],
        "damages": [{"damage_type": "scratch", "part_type": "hood"}],
    }

    await consumer._handle(message)

    assert len(use_case.calls) == 1
    received = use_case.calls[0].data
    assert received.parts[0].confidence == 0.0
    assert received.damages[0].confidence == 0.0


@pytest.mark.anyio
async def test_missing_status_field_defaults_to_error() -> None:
    # Reading the source: ``message.get("status", "error")``. A missing status
    # must be treated as failure so the user falls back to manual mode rather
    # than silently receiving an "empty success" response.
    use_case = _StubUseCase()
    consumer = _consumer(use_case)
    message: dict[str, Any] = {
        "request_id": "req-3",
        "parts": [],
        "damages": [],
    }

    await consumer._handle(message)

    assert use_case.calls[0].data.status == "error"
