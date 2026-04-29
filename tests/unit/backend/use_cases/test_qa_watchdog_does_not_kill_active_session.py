"""QA: watchdog убивает активную сессию пользователя.

Сценарий 1 из задания: заявка создана в T=0 (timeout_at = T+5мин). Пользователь
спокойно выбирает детали, на T+4мин добавляет ещё одно повреждение, на T+5мин10с
срабатывает HeartbeatChecker — заявка должна оставаться активной, потому что
пользователь проявил активность.

Текущий код:
* ``RepairRequest.new`` фиксирует ``timeout_at = created_at + 5 минут`` и никогда
  его не обновляет (``with_status`` копирует ``timeout_at`` без изменений).
* Ни один use-case (``AddDamageUseCase``, ``EditDamageUseCase``,
  ``DeleteDamageUseCase``, ``UploadPhotoUseCase``) не продлевает ``timeout_at``.
* ``HeartbeatChecker._check_timeouts`` берёт любой request с
  ``timeout_at <= now()`` и переводит в FAILED.

Итог: любая заявка, с которой пользователь возится дольше 5 минут, будет убита
watchdog-ом даже если он активно нажимает кнопки.

Эти тесты зашивают правильный контракт: каждое пользовательское действие
обязано перенести ``timeout_at`` на "сейчас + 5 минут". Единица времени
"5 минут" читается из ``RepairRequest.new`` (там она hard-coded) — тесты
используют гибкую границу `>= now + 4 мин`, чтобы не навязывать точную
величину, но поймать отсутствие продления.
"""

from __future__ import annotations

import datetime as _dt
from uuid import uuid4

import pytest

from auto_repair_estimator.backend.adapters.repositories.in_memory_outbox_repository import (
    InMemoryOutboxRepository,
)
from auto_repair_estimator.backend.adapters.repositories.in_memory_repair_request_repository import (
    InMemoryRepairRequestRepository,
)
from auto_repair_estimator.backend.domain.entities.detected_damage import DetectedDamage
from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.services.request_state_machine import RequestStateMachine
from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    DamageSource,
    DamageType,
    PartType,
    RequestMode,
    RequestStatus,
)
from auto_repair_estimator.backend.use_cases.manage_damages import (
    AddDamageInput,
    AddDamageUseCase,
    DeleteDamageInput,
    DeleteDamageUseCase,
    EditDamageInput,
    EditDamageUseCase,
)
from auto_repair_estimator.backend.use_cases.repair_requests import (
    UploadPhotoInput,
    UploadPhotoUseCase,
)

# How close to the new baseline (now) the extended timeout_at must be.
# We give a 1-minute slack so that clock drift and test execution time do
# not produce false positives, while still catching the "timeout_at was
# never updated" bug (where the gap would be 0-4 minutes depending on the
# simulated user activity time).
_MIN_EXTENSION_FROM_NOW = _dt.timedelta(minutes=4)


class _FakeDamageRepo:
    def __init__(self) -> None:
        self._items: dict[str, DetectedDamage] = {}

    async def add(self, damage: DetectedDamage) -> None:
        self._items[damage.id] = damage

    async def get(self, damage_id: str) -> DetectedDamage | None:
        return self._items.get(damage_id)

    async def get_by_request_id(self, request_id: str) -> list[DetectedDamage]:
        return [d for d in self._items.values() if d.request_id == request_id]

    async def update(self, damage: DetectedDamage) -> None:
        self._items[damage.id] = damage

    async def soft_delete(self, damage_id: str) -> None:
        if damage_id in self._items:
            d = self._items[damage_id]
            self._items[damage_id] = DetectedDamage(
                id=d.id,
                request_id=d.request_id,
                damage_type=d.damage_type,
                part_type=d.part_type,
                source=d.source,
                is_deleted=True,
                part_id=d.part_id,
                confidence=d.confidence,
                mask_image_key=d.mask_image_key,
            )


async def _seed_near_timeout_request(
    request_repo: InMemoryRepairRequestRepository,
    *,
    status: RequestStatus,
    mode: RequestMode,
    minutes_since_create: int = 4,
) -> RepairRequest:
    """Create a request that was made `minutes_since_create` minutes ago,
    so it is close to (but not yet past) the 5-minute watchdog deadline.
    """

    created_at = _dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=minutes_since_create)
    request = RepairRequest(
        id=str(uuid4()),
        chat_id=100,
        user_id=200,
        mode=mode,
        status=status,
        created_at=created_at,
        updated_at=created_at,
        timeout_at=created_at + _dt.timedelta(minutes=5),
    )
    await request_repo.add(request)
    return request


def _assert_timeout_was_extended(
    before: RepairRequest,
    after: RepairRequest,
    action: str,
) -> None:
    """Central assertion: after a user action the new timeout_at must sit
    at least ~4 minutes in the future, otherwise the watchdog will kill
    the session within one heartbeat interval."""

    now = _dt.datetime.now(_dt.UTC)
    min_allowed = now + _MIN_EXTENSION_FROM_NOW
    assert after.timeout_at >= min_allowed, (
        f"After {action} timeout_at should have been extended to "
        f">= {min_allowed.isoformat()} (now + 4 min), "
        f"but it is {after.timeout_at.isoformat()}. "
        f"Before the action it was {before.timeout_at.isoformat()}. "
        "Watchdog will kill this request within one heartbeat interval "
        "even though the user is actively interacting."
    )


@pytest.mark.anyio
async def test_adding_damage_extends_timeout_on_near_timeout_request() -> None:
    """Сценарий 1: пользователь 4 минуты выбирал повреждение и добавил его.
    После add_damage timeout_at должен сдвинуться в будущее.
    """

    # Arrange
    req_repo = InMemoryRepairRequestRepository()
    dmg_repo = _FakeDamageRepo()
    request = await _seed_near_timeout_request(
        req_repo, status=RequestStatus.PRICING, mode=RequestMode.MANUAL
    )
    use_case = AddDamageUseCase(request_repository=req_repo, damage_repository=dmg_repo)

    # Act — user adds a damage after 4 minutes of deliberation.
    await use_case.execute(
        AddDamageInput(
            request_id=request.id,
            part_type=PartType.HOOD,
            damage_type=DamageType.DENT,
        )
    )

    # Assert
    after = await req_repo.get(request.id)
    assert after is not None
    _assert_timeout_was_extended(request, after, action="AddDamageUseCase")


@pytest.mark.anyio
async def test_editing_damage_extends_timeout_on_near_timeout_request() -> None:
    """Изменение типа/детали повреждения — тоже активность пользователя."""

    # Arrange
    req_repo = InMemoryRepairRequestRepository()
    dmg_repo = _FakeDamageRepo()
    request = await _seed_near_timeout_request(
        req_repo, status=RequestStatus.PRICING, mode=RequestMode.MANUAL
    )
    existing = DetectedDamage(
        id=str(uuid4()),
        request_id=request.id,
        damage_type=DamageType.SCRATCH,
        part_type=PartType.DOOR,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    await dmg_repo.add(existing)

    use_case = EditDamageUseCase(damage_repository=dmg_repo, request_repository=req_repo)

    # Act — user switches the damage type 4 minutes in.
    await use_case.execute(
        EditDamageInput(damage_id=existing.id, damage_type=DamageType.DENT)
    )

    # Assert
    after = await req_repo.get(request.id)
    assert after is not None
    _assert_timeout_was_extended(request, after, action="EditDamageUseCase")


@pytest.mark.anyio
async def test_deleting_damage_extends_timeout_on_near_timeout_request() -> None:
    """Удаление одного из ошибочно добавленных повреждений — активность."""

    # Arrange
    req_repo = InMemoryRepairRequestRepository()
    dmg_repo = _FakeDamageRepo()
    request = await _seed_near_timeout_request(
        req_repo, status=RequestStatus.PRICING, mode=RequestMode.MANUAL
    )
    existing = DetectedDamage(
        id=str(uuid4()),
        request_id=request.id,
        damage_type=DamageType.SCRATCH,
        part_type=PartType.DOOR,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    await dmg_repo.add(existing)
    use_case = DeleteDamageUseCase(damage_repository=dmg_repo, request_repository=req_repo)

    # Act
    await use_case.execute(DeleteDamageInput(damage_id=existing.id))

    # Assert
    after = await req_repo.get(request.id)
    assert after is not None
    _assert_timeout_was_extended(request, after, action="DeleteDamageUseCase")


@pytest.mark.anyio
async def test_uploading_photo_extends_timeout_on_near_timeout_request() -> None:
    """ML flow: пользователь долго искал фото и прислал его на 4-й минуте."""

    # Arrange
    req_repo = InMemoryRepairRequestRepository()
    outbox_repo = InMemoryOutboxRepository()
    request = await _seed_near_timeout_request(
        req_repo, status=RequestStatus.CREATED, mode=RequestMode.ML
    )
    use_case = UploadPhotoUseCase(
        repository=req_repo,
        state_machine=RequestStateMachine(),
        outbox_repository=outbox_repo,
        inference_requests_topic="inference_requests",
    )

    # Act
    await use_case.execute(UploadPhotoInput(request_id=request.id, image_key="raw/1.jpg"))

    # Assert
    after = await req_repo.get(request.id)
    assert after is not None
    _assert_timeout_was_extended(request, after, action="UploadPhotoUseCase")


@pytest.mark.anyio
async def test_state_transition_extends_timeout() -> None:
    """Перевод статуса через ``RepairRequest.with_status`` (используется в
    use-case'ах и в state_machine) — это, по сути, тоже активность
    системы по конкретной заявке и должен продлевать watchdog.

    Без этого любая долгая ML-обработка или переход QUEUED -> PROCESSING
    -> PRICING не защищены от watchdog-а, если суммарно заняли > 5 минут
    (в проде это норма для очереди)."""

    now = _dt.datetime.now(_dt.UTC)
    created_at = now - _dt.timedelta(minutes=4)
    request = RepairRequest(
        id=str(uuid4()),
        chat_id=1,
        user_id=2,
        mode=RequestMode.ML,
        status=RequestStatus.QUEUED,
        created_at=created_at,
        updated_at=created_at,
        timeout_at=created_at + _dt.timedelta(minutes=5),
    )

    # Act — state transition.
    transitioned = request.with_status(RequestStatus.PROCESSING)

    # Assert
    _assert_timeout_was_extended(request, transitioned, action="RepairRequest.with_status")
