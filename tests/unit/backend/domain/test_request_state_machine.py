from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pytest import mark, raises

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.services.request_state_machine import (
    InvalidStatusTransitionError,
    RequestStateMachine,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestMode, RequestStatus


def _new_request(mode: RequestMode, status: RequestStatus | None = None) -> RepairRequest:
    request_id = str(uuid4())
    now = datetime.now(UTC)
    timeout_at = now + timedelta(minutes=5)
    if status is None:
        status = RequestStatus.PRICING if mode is RequestMode.MANUAL else RequestStatus.CREATED
    return RepairRequest(
        id=request_id,
        chat_id=1,
        user_id=2,
        mode=mode,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=timeout_at,
    )


@mark.parametrize(
    "from_status,to_status",
    [
        (RequestStatus.CREATED, RequestStatus.QUEUED),
        (RequestStatus.CREATED, RequestStatus.PRICING),
        (RequestStatus.CREATED, RequestStatus.FAILED),
        (RequestStatus.QUEUED, RequestStatus.PROCESSING),
        (RequestStatus.QUEUED, RequestStatus.FAILED),
        (RequestStatus.PROCESSING, RequestStatus.PRICING),
        (RequestStatus.PROCESSING, RequestStatus.FAILED),
        (RequestStatus.PRICING, RequestStatus.DONE),
        (RequestStatus.PRICING, RequestStatus.FAILED),
    ],
)
def test_can_transition_allows_valid_edges(from_status: RequestStatus, to_status: RequestStatus) -> None:
    sm = RequestStateMachine()
    assert sm.can_transition(from_status, to_status) is True


@mark.parametrize(
    "from_status,to_status",
    [
        (RequestStatus.DONE, RequestStatus.PRICING),
        (RequestStatus.DONE, RequestStatus.FAILED),
        (RequestStatus.FAILED, RequestStatus.PRICING),
        (RequestStatus.FAILED, RequestStatus.DONE),
        (RequestStatus.QUEUED, RequestStatus.DONE),
        (RequestStatus.CREATED, RequestStatus.DONE),
        (RequestStatus.PRICING, RequestStatus.CREATED),
        (RequestStatus.PROCESSING, RequestStatus.CREATED),
    ],
)
def test_can_transition_rejects_invalid_edges(from_status: RequestStatus, to_status: RequestStatus) -> None:
    sm = RequestStateMachine()
    assert sm.can_transition(from_status, to_status) is False


@mark.parametrize(
    "initial_mode,expected_status",
    [
        (RequestMode.ML, RequestStatus.CREATED),
        (RequestMode.MANUAL, RequestStatus.PRICING),
    ],
)
def test_repair_request_new_sets_initial_status(initial_mode: RequestMode, expected_status: RequestStatus) -> None:
    request = RepairRequest.new(request_id=str(uuid4()), chat_id=1, user_id=2, mode=initial_mode)
    assert request.mode is initial_mode
    assert request.status is expected_status
    assert request.timeout_at > request.created_at


def test_transition_updates_status_and_timestamp() -> None:
    sm = RequestStateMachine()
    request = _new_request(mode=RequestMode.ML, status=RequestStatus.CREATED)
    next_request = sm.transition(request, RequestStatus.QUEUED)
    assert next_request.status is RequestStatus.QUEUED
    assert next_request.updated_at > request.updated_at
    assert next_request.id == request.id


def test_transition_same_status_is_noop() -> None:
    sm = RequestStateMachine()
    request = _new_request(mode=RequestMode.ML, status=RequestStatus.QUEUED)
    next_request = sm.transition(request, RequestStatus.QUEUED)
    assert next_request is request


def test_transition_invalid_raises_error() -> None:
    sm = RequestStateMachine()
    request = _new_request(mode=RequestMode.ML, status=RequestStatus.DONE)
    with raises(InvalidStatusTransitionError) as exc:
        sm.transition(request, RequestStatus.PRICING)
    assert exc.value.from_status is RequestStatus.DONE
    assert exc.value.to_status is RequestStatus.PRICING
