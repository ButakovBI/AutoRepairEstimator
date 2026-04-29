"""QA: RepairRequest.new must produce timezone-aware UTC timestamps.

Postgres ``TIMESTAMPTZ`` columns require tz-aware Python datetimes. If a
naive ``datetime.now()`` ever sneaks in, asyncpg will treat the time as
local-server-time (depending on the asyncpg connection settings), which
silently shifts ``timeout_at`` by a few hours depending on deployment
locale — masking bugs until production traffic hits a timezone boundary.

The entity is the single source of truth for freshly created requests;
verify that guarantee here rather than in every callsite.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus


def test_created_at_is_timezone_aware_and_in_utc() -> None:
    request = RepairRequest.new(
        request_id="r-1", chat_id=1, user_id=2, mode=RequestMode.MANUAL
    )
    assert request.created_at.tzinfo is not None, (
        "RepairRequest.created_at is naive. Postgres TIMESTAMPTZ requires a "
        "tz-aware datetime; a naive one would be inserted as 'local' and shift "
        "silently between the test machine and the production container."
    )
    # utcoffset() == 0 is the only reliable UTC check; ``tzinfo is UTC`` is
    # not guaranteed because asyncpg may round-trip through another UTC alias.
    assert request.created_at.utcoffset() == timedelta(0)


def test_timeout_at_is_after_created_at_and_tz_aware() -> None:
    request = RepairRequest.new(
        request_id="r-1", chat_id=1, user_id=2, mode=RequestMode.ML
    )
    assert request.timeout_at.tzinfo is not None
    assert request.timeout_at > request.created_at, (
        "timeout_at must strictly follow created_at — otherwise HeartbeatChecker "
        "would flag every fresh request as already expired."
    )


def test_manual_mode_starts_in_pricing_status() -> None:
    """Business rule from the spec: manual requests skip ML queue entirely."""
    request = RepairRequest.new(
        request_id="r-1", chat_id=1, user_id=2, mode=RequestMode.MANUAL
    )
    assert request.status is RequestStatus.PRICING, (
        "Manual mode must start in PRICING — otherwise the user cannot add "
        "damages immediately after /start."
    )


def test_ml_mode_starts_in_created_status() -> None:
    """ML flow must start in CREATED so the upload-photo use case can drive
    it to QUEUED. Starting in any other state would either skip the photo
    step or stop the workflow."""
    request = RepairRequest.new(
        request_id="r-1", chat_id=1, user_id=2, mode=RequestMode.ML
    )
    assert request.status is RequestStatus.CREATED


def test_timeout_window_is_reasonable_for_user_interaction() -> None:
    """The exact timeout is a product decision, but the envelope must be
    sane: a user must have at least one minute (to decide/edit/confirm)
    and at most one hour (otherwise failed ML sessions linger forever and
    the heartbeat worker does nothing useful)."""
    request = RepairRequest.new(
        request_id="r-1", chat_id=1, user_id=2, mode=RequestMode.ML
    )
    delta = request.timeout_at - request.created_at
    assert timedelta(minutes=1) <= delta <= timedelta(hours=1), (
        f"Suspicious initial timeout window: {delta}. The value is outside the "
        "safe 1 min .. 1 h envelope — either the heartbeat will trigger "
        "immediately or it will never fire."
    )
