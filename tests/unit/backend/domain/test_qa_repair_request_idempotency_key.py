"""QA: idempotency_key support is declared by the plan but not implemented.

The architectural plan is explicit in two places:

* In the database schema (``docker/init.sql``) ``repair_requests`` carries
  ``idempotency_key VARCHAR UNIQUE``.
* In the "Key Design Decisions" section: *"idempotency_key (chat_id:message_id)
  on repair_requests for дедупликации фото"* — the key should uniquely
  identify the VK attachment that triggered the request, so redelivered
  VK events don't create duplicate ``RepairRequest`` rows.

In the current code neither layer wires it up:

* ``RepairRequest`` (the domain entity) does not expose ``idempotency_key``.
* ``PostgresRepairRequestRepository.add`` hard-codes ``None`` for the column.

As a consequence the ``UNIQUE`` constraint in Postgres is useless (every row
has a NULL idempotency_key) and the dedup feature can never fire. This
test file encodes the missing contract so the fix is specification-driven.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt

import pytest

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode


def test_repair_request_entity_exposes_idempotency_key_field() -> None:
    """The entity must model the field that the SQL schema already reserves."""

    field_names = {f.name for f in dataclasses.fields(RepairRequest)}
    assert "idempotency_key" in field_names, (
        "RepairRequest is missing the 'idempotency_key' field that init.sql "
        "already declares with a UNIQUE constraint. Without this field the "
        "domain layer cannot enforce photo-level deduplication required by "
        "the plan's 'Key Design Decisions' section."
    )


def test_repair_request_new_accepts_idempotency_key_kwarg() -> None:
    """``RepairRequest.new`` is the only constructor used by the create-request
    use case; it must let callers pass an idempotency key so that the bot's
    `chat_id:message_id` composite can be attached at creation time."""

    try:
        request = RepairRequest.new(
            request_id="r-1",
            chat_id=10,
            user_id=20,
            mode=RequestMode.ML,
            idempotency_key="10:100",  # type: ignore[call-arg]
        )
    except TypeError as exc:
        pytest.fail(
            "RepairRequest.new does not accept an 'idempotency_key' kwarg. "
            "The create-request use case therefore cannot persist the dedup "
            f"token coming from the bot. Underlying error: {exc}"
        )

    assert getattr(request, "idempotency_key", None) == "10:100", (
        "RepairRequest.new accepted the kwarg but did not store it on the entity."
    )


def test_repair_request_with_status_preserves_idempotency_key() -> None:
    """State-machine transitions must not accidentally drop the dedup key."""

    try:
        request = RepairRequest.new(
            request_id="r-1",
            chat_id=10,
            user_id=20,
            mode=RequestMode.ML,
            idempotency_key="10:100",  # type: ignore[call-arg]
        )
    except TypeError:
        pytest.skip("Covered by test_repair_request_new_accepts_idempotency_key_kwarg")

    # Use with_status to simulate a state transition.
    from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestStatus

    transitioned = request.with_status(
        RequestStatus.QUEUED,
        updated_at=_dt.datetime.now(_dt.UTC),
    )
    assert getattr(transitioned, "idempotency_key", None) == "10:100", (
        "RepairRequest.with_status dropped the idempotency_key. Every subsequent "
        "state update would then lose the dedup token, breaking retry semantics."
    )
