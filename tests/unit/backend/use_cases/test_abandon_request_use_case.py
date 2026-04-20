"""Tests for ``AbandonRequestUseCase``.

Business contract:

* Non-terminal statuses (CREATED/QUEUED/PROCESSING/PRICING) all transition
  to FAILED when the user or the bot explicitly abandons the session.
  This is what backs the "press 'Начать' while a request is still active"
  UX fix — without it, a second ``CreateRepairRequestUseCase`` call
  leaves two non-terminal rows for the same ``chat_id`` and
  ``get_latest_active_by_chat_id`` starts returning whichever was
  inserted last, making stale buttons mutate the wrong session.

* Already-terminal statuses (DONE/FAILED) must NOT error — the endpoint
  is idempotent so the bot can always call it before starting a new
  session without branching on the current state. The result just
  signals ``was_terminal=True`` so the caller can log / metric it.

* The cause string (default ``"user_abandoned"``) lands in
  ``ml_error_code`` so ops can distinguish user-initiated abandons from
  ML failures and watchdog timeouts in logs / dashboards.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from auto_repair_estimator.backend.adapters.repositories.in_memory_repair_request_repository import (
    InMemoryRepairRequestRepository,
)
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    RequestMode,
    RequestStatus,
)
from auto_repair_estimator.backend.use_cases.repair_requests import (
    AbandonRequestInput,
    AbandonRequestUseCase,
)


def _make_request(status: RequestStatus, mode: RequestMode = RequestMode.ML) -> RepairRequest:
    now = datetime.now(UTC)
    return RepairRequest(
        id=str(uuid4()),
        chat_id=1,
        user_id=2,
        mode=mode,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=now + timedelta(minutes=5),
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status",
    [
        RequestStatus.CREATED,
        RequestStatus.QUEUED,
        RequestStatus.PROCESSING,
        RequestStatus.PRICING,
    ],
)
async def test_abandon_transitions_non_terminal_to_failed(status: RequestStatus) -> None:
    """Every non-terminal status must land in FAILED and stamp the reason."""
    repo = InMemoryRepairRequestRepository()
    use_case = AbandonRequestUseCase(repository=repo, state_machine=RequestStateMachine())
    request = _make_request(status)
    await repo.add(request)

    result = await use_case.execute(AbandonRequestInput(request_id=request.id))

    assert result.was_terminal is False
    assert result.request.status is RequestStatus.FAILED
    # Cause code must be present so operators can tell apart user-abandon
    # from ML-failure and watchdog-timeout rows.
    assert result.request.ml_error_code == "user_abandoned"
    # Persisted state matches the result we returned.
    stored = await repo.get(request.id)
    assert stored is not None
    assert stored.status is RequestStatus.FAILED
    assert stored.ml_error_code == "user_abandoned"


@pytest.mark.anyio
@pytest.mark.parametrize("status", [RequestStatus.DONE, RequestStatus.FAILED])
async def test_abandon_is_idempotent_for_terminal_statuses(status: RequestStatus) -> None:
    """Already-terminal requests are a no-op — the bot calls this
    unconditionally before starting a new session and must not error."""
    repo = InMemoryRepairRequestRepository()
    use_case = AbandonRequestUseCase(repository=repo, state_machine=RequestStateMachine())
    request = _make_request(status)
    # Simulate a prior ml_error_code that must NOT be overwritten, so a
    # later retry of abandon() doesn't rewrite historic failure reasons.
    request.ml_error_code = "ml_inference_error"
    await repo.add(request)

    result = await use_case.execute(AbandonRequestInput(request_id=request.id))

    assert result.was_terminal is True
    assert result.request.status is status
    assert result.request.ml_error_code == "ml_inference_error"


@pytest.mark.anyio
async def test_abandon_missing_request_raises_value_error() -> None:
    repo = InMemoryRepairRequestRepository()
    use_case = AbandonRequestUseCase(repository=repo, state_machine=RequestStateMachine())

    with pytest.raises(ValueError, match="not found"):
        await use_case.execute(AbandonRequestInput(request_id="does-not-exist"))


@pytest.mark.anyio
async def test_abandon_reason_override_is_recorded() -> None:
    """Callers (bot, ops tooling) can pass their own reason — useful for
    separating "user clicked Начать" from other explicit abandons later."""
    repo = InMemoryRepairRequestRepository()
    use_case = AbandonRequestUseCase(repository=repo, state_machine=RequestStateMachine())
    request = _make_request(RequestStatus.PRICING)
    await repo.add(request)

    result = await use_case.execute(
        AbandonRequestInput(request_id=request.id, reason="mode_switch")
    )

    assert result.request.ml_error_code == "mode_switch"
