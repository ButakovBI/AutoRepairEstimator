"""QA: каждая мутация заявки должна строго фильтровать текущий статус.

Сценарий 2 из задания: после DONE (или до Начать) кнопки старых клавиатур
VK остаются у пользователя. Нажатие на них триггерит callback, который
сейчас бежит в backend и возвращает «нормальный ответ», хотя должен был
отказать, потому что заявка уже закрыта / ещё не в нужном статусе.

Правильный state-контракт (из плана и state-machine):

* add_damage           — разрешён ТОЛЬКО в PRICING
* edit_damage          — разрешён ТОЛЬКО в PRICING
* delete_damage        — разрешён ТОЛЬКО в PRICING
* upload_photo         — разрешён ТОЛЬКО в CREATED (ML mode) [уже работает]
* confirm_pricing      — разрешён ТОЛЬКО в PRICING [уже работает]

Тесты прогоняются параметрически по всем нерелевантным статусам и проверяют,
что use-case поднимает ``ValueError``. Это тот же контракт, что и
``ConfirmPricingUseCase`` / ``UploadPhotoUseCase`` уже соблюдают — его надо
распространить на damage-мутации.
"""

from __future__ import annotations

import datetime as _dt
from uuid import uuid4

import pytest

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
    ConfirmPricingInput,
    ConfirmPricingUseCase,
    UploadPhotoInput,
    UploadPhotoUseCase,
)

# Статусы, при которых мутация повреждений НЕ должна проходить (всё, что ≠ PRICING).
_NON_PRICING_STATUSES = [
    RequestStatus.CREATED,
    RequestStatus.QUEUED,
    RequestStatus.PROCESSING,
    RequestStatus.DONE,
    RequestStatus.FAILED,
]


class _FakeRequestRepo:
    def __init__(self) -> None:
        self._items: dict[str, RepairRequest] = {}

    async def add(self, request: RepairRequest) -> None:
        self._items[request.id] = request

    async def get(self, request_id: str) -> RepairRequest | None:
        return self._items.get(request_id)

    async def update(self, request: RepairRequest) -> None:
        self._items[request.id] = request

    async def get_timed_out_requests(self):  # type: ignore[no-untyped-def]
        return []


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
            )


class _FakeOutboxRepo:
    def __init__(self) -> None:
        self._items: list[object] = []

    async def add(self, event: object) -> None:
        self._items.append(event)

    async def get_unpublished(self, limit: int) -> list[object]:
        return []

    async def mark_published(self, ids: list[str]) -> None:
        return None


def _make_request(status: RequestStatus, mode: RequestMode = RequestMode.MANUAL) -> RepairRequest:
    now = _dt.datetime.now(_dt.UTC)
    return RepairRequest(
        id=str(uuid4()),
        chat_id=1,
        user_id=2,
        mode=mode,
        status=status,
        created_at=now,
        updated_at=now,
        timeout_at=now + _dt.timedelta(minutes=5),
    )


# ---------------------------------------------------------------------------
# add_damage: allowed only in PRICING
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("status", _NON_PRICING_STATUSES, ids=lambda s: s.value)
async def test_add_damage_is_rejected_outside_pricing_status(status: RequestStatus) -> None:
    """Например, пользователь ответил на завершённое ML-сообщение стеклой
    клавиатурой ручного режима — add_damage должен упасть с 400.
    """

    req_repo = _FakeRequestRepo()
    req = _make_request(status)
    await req_repo.add(req)
    use_case = AddDamageUseCase(request_repository=req_repo, damage_repository=_FakeDamageRepo())

    with pytest.raises(ValueError):
        await use_case.execute(
            AddDamageInput(
                request_id=req.id,
                part_type=PartType.HOOD,
                damage_type=DamageType.DENT,
            )
        )


# ---------------------------------------------------------------------------
# edit_damage: allowed only in PRICING
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("status", _NON_PRICING_STATUSES, ids=lambda s: s.value)
async def test_edit_damage_is_rejected_outside_pricing_status(status: RequestStatus) -> None:
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req = _make_request(status)
    await req_repo.add(req)
    dmg = DetectedDamage(
        id=str(uuid4()),
        request_id=req.id,
        damage_type=DamageType.SCRATCH,
        part_type=PartType.HOOD,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    await dmg_repo.add(dmg)

    use_case = EditDamageUseCase(damage_repository=dmg_repo)
    # Attach the request repo through the attribute the fix is expected to use.
    use_case._requests = req_repo  # type: ignore[attr-defined]  # noqa: SLF001

    with pytest.raises(ValueError):
        await use_case.execute(EditDamageInput(damage_id=dmg.id, damage_type=DamageType.DENT))


# ---------------------------------------------------------------------------
# delete_damage: allowed only in PRICING
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("status", _NON_PRICING_STATUSES, ids=lambda s: s.value)
async def test_delete_damage_is_rejected_outside_pricing_status(status: RequestStatus) -> None:
    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req = _make_request(status)
    await req_repo.add(req)
    dmg = DetectedDamage(
        id=str(uuid4()),
        request_id=req.id,
        damage_type=DamageType.SCRATCH,
        part_type=PartType.HOOD,
        source=DamageSource.MANUAL,
        is_deleted=False,
    )
    await dmg_repo.add(dmg)

    use_case = DeleteDamageUseCase(damage_repository=dmg_repo)
    use_case._requests = req_repo  # type: ignore[attr-defined]  # noqa: SLF001

    with pytest.raises(ValueError):
        await use_case.execute(DeleteDamageInput(damage_id=dmg.id))


# ---------------------------------------------------------------------------
# upload_photo and confirm_pricing already gate correctly —
# lock in the contract so a future refactor doesn't regress it.
# ---------------------------------------------------------------------------


_UPLOAD_BAD_STATUSES = [
    RequestStatus.QUEUED,
    RequestStatus.PROCESSING,
    RequestStatus.PRICING,
    RequestStatus.DONE,
    RequestStatus.FAILED,
]


@pytest.mark.anyio
@pytest.mark.parametrize("status", _UPLOAD_BAD_STATUSES, ids=lambda s: s.value)
async def test_upload_photo_is_rejected_outside_created_status(status: RequestStatus) -> None:
    req_repo = _FakeRequestRepo()
    outbox = _FakeOutboxRepo()
    req = _make_request(status, mode=RequestMode.ML)
    await req_repo.add(req)
    use_case = UploadPhotoUseCase(
        repository=req_repo,
        state_machine=RequestStateMachine(),
        outbox_repository=outbox,
        inference_requests_topic="inference_requests",
    )

    with pytest.raises(ValueError):
        await use_case.execute(UploadPhotoInput(request_id=req.id, image_key="raw/x.jpg"))


_CONFIRM_BAD_STATUSES = [
    RequestStatus.CREATED,
    RequestStatus.QUEUED,
    RequestStatus.PROCESSING,
    RequestStatus.DONE,
    RequestStatus.FAILED,
]


@pytest.mark.anyio
@pytest.mark.parametrize("status", _CONFIRM_BAD_STATUSES, ids=lambda s: s.value)
async def test_confirm_pricing_is_rejected_outside_pricing_status(status: RequestStatus) -> None:
    """Самое очевидное проявление сценария 2: после DONE у пользователя
    остаётся кнопка "Подтвердить", и она не должна ещё раз переводить
    заявку в DONE или поднимать её из FAILED."""

    req_repo = _FakeRequestRepo()
    req = _make_request(status)
    await req_repo.add(req)
    use_case = ConfirmPricingUseCase(repository=req_repo, state_machine=RequestStateMachine())

    with pytest.raises(ValueError):
        await use_case.execute(ConfirmPricingInput(request_id=req.id))


# ---------------------------------------------------------------------------
# End-to-end stale-button scenario: PRICING -> DONE, then a stale add_damage
# must be rejected even if the damage happens to exist in the DB.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_stale_button_after_done_does_not_resurrect_the_request() -> None:
    """Сценарий: пользователь завершил заявку (DONE), но не нажал /start.
    Нажатие на старую кнопку "Добавить повреждение" → part → damage_type
    не должно тихо доводить мутацию до конца — ответ backend-а обязан быть
    4xx, чтобы бот показал понятное сообщение, а не "Добавлено: …".
    """

    req_repo = _FakeRequestRepo()
    dmg_repo = _FakeDamageRepo()
    req = _make_request(RequestStatus.DONE, mode=RequestMode.MANUAL)
    await req_repo.add(req)

    use_case = AddDamageUseCase(request_repository=req_repo, damage_repository=dmg_repo)

    with pytest.raises(ValueError):
        await use_case.execute(
            AddDamageInput(
                request_id=req.id,
                part_type=PartType.BUMPER,
                damage_type=DamageType.DENT,
            )
        )
    # И в БД повреждений заявки не должно появиться ничего.
    assert await dmg_repo.get_by_request_id(req.id) == [], (
        "AddDamageUseCase всё же записал повреждение в закрытую заявку, "
        "пользователь получит 'нормальный ответ' на stale-кнопку."
    )
