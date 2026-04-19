"""Adversarial tests pinning the terminal-state semantics of the state
machine and the monotonic timestamp guarantee.

Gaps found in the existing coverage:

* ``transition()`` from ``FAILED`` was never exercised (only ``can_transition``
  returned False). A future refactor could accidentally permit it.
* The monotonic ``updated_at`` guarantee (line ``if updated_at <= request
  .updated_at: updated_at = request.updated_at + 1us``) is implemented but
  untested. Without it, two transitions fired in the same microsecond would
  look like "no progress" to time-based consumers.
* ``can_transition`` self-loop behaviour: ``QUEUED -> QUEUED`` is True; the
  associated ``transition`` call must return the SAME instance (no timestamp
  bump), because an idempotent retry shouldn't reset the clock.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.services.request_state_machine import (
    InvalidStatusTransitionError,
    RequestStateMachine,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    RequestMode,
    RequestStatus,
)


def _req(status: RequestStatus, *, updated_at: datetime | None = None) -> RepairRequest:
    now = datetime.now(UTC)
    ts = updated_at or now
    return RepairRequest(
        id="req-1",
        chat_id=1,
        user_id=2,
        mode=RequestMode.ML,
        status=status,
        created_at=now,
        updated_at=ts,
        timeout_at=now,
    )


@pytest.mark.parametrize(
    "target",
    [
        RequestStatus.CREATED,
        RequestStatus.QUEUED,
        RequestStatus.PROCESSING,
        RequestStatus.PRICING,
        RequestStatus.DONE,
    ],
)
def test_transition_from_failed_to_any_other_state_raises(target: RequestStatus) -> None:
    sm = RequestStateMachine()

    with pytest.raises(InvalidStatusTransitionError) as exc_info:
        sm.transition(_req(RequestStatus.FAILED), target)

    assert exc_info.value.from_status is RequestStatus.FAILED
    assert exc_info.value.to_status is target


@pytest.mark.parametrize(
    "target",
    [
        RequestStatus.CREATED,
        RequestStatus.QUEUED,
        RequestStatus.PROCESSING,
        RequestStatus.PRICING,
        RequestStatus.FAILED,
    ],
)
def test_transition_from_done_to_any_other_state_raises(target: RequestStatus) -> None:
    sm = RequestStateMachine()

    with pytest.raises(InvalidStatusTransitionError) as exc_info:
        sm.transition(_req(RequestStatus.DONE), target)

    assert exc_info.value.from_status is RequestStatus.DONE


def test_self_transition_returns_the_same_request_instance_without_touching_timestamps() -> None:
    sm = RequestStateMachine()
    original_ts = datetime.now(UTC).replace(microsecond=123456)
    r = _req(RequestStatus.QUEUED, updated_at=original_ts)

    result = sm.transition(r, RequestStatus.QUEUED)

    # Same object — we explicitly short-circuit self-transitions in the source.
    assert result is r
    assert result.updated_at == original_ts


def test_consecutive_transitions_produce_strictly_monotonic_updated_at() -> None:
    """Two transitions can legitimately be requested within the same
    microsecond (fast tests, high-frequency state machines). ``updated_at``
    must still strictly increase so downstream time-ordered consumers (e.g.
    audit log, heartbeat checker) can tell them apart.
    """

    sm = RequestStateMachine()
    # Seed with a very near-future timestamp so ``datetime.now`` on the next
    # transition call is likely to be <= than it.
    seeded_ts = datetime.now(UTC).replace(microsecond=999999)
    r = _req(RequestStatus.QUEUED, updated_at=seeded_ts)

    r2 = sm.transition(r, RequestStatus.PROCESSING)

    assert r2.updated_at > r.updated_at, (
        "State machine must guarantee strictly-monotonic updated_at; instead "
        f"got {r2.updated_at.isoformat()} <= {r.updated_at.isoformat()}."
    )


def test_error_message_on_invalid_transition_names_both_ends() -> None:
    """An operator reading the log must be able to tell which transition
    was rejected without diving into the exception class. The current
    implementation formats ``from.value -> to.value``; future refactors
    that drop either side would degrade on-call observability.
    """

    sm = RequestStateMachine()
    with pytest.raises(InvalidStatusTransitionError) as exc_info:
        sm.transition(_req(RequestStatus.DONE), RequestStatus.PRICING)

    assert "done" in str(exc_info.value)
    assert "pricing" in str(exc_info.value)
