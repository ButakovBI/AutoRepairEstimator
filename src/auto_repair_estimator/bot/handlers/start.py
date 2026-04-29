from __future__ import annotations

from typing import Any

from vkbottle import API
from vkbottle.bot import Message, MessageEvent

from auto_repair_estimator.bot.backend_client import BackendClient
from auto_repair_estimator.bot.keyboards.mode_selection import mode_selection_keyboard
from auto_repair_estimator.bot.keyboards.start import start_keyboard
from auto_repair_estimator.bot.session_lifecycle import (
    PREVIOUS_REQUEST_ABANDONED_NOTICE,
    abandon_active_session,
)

WELCOME_TEXT = (
    "Добро пожаловать! Я помогу оценить стоимость ремонта вашего автомобиля.\n\n"
    "Для этого вы можете:\n"
    "- Загрузить фотографию повреждения — я проанализирую его с помощью AI\n"
    "- Указать повреждения вручную — выбрав деталь и тип повреждения\n\n"
    "Выберите режим работы:"
)

# The reject message is the *single* cross-session nudge the user receives
# whenever they try to interact with the bot outside an active scenario —
# free-text "hello", a stale button from an expired session, a retry of a
# terminal callback. Kept as a module constant so every call site renders
# exactly the same wording (QA relies on this for contract assertions).
NO_ACTIVE_SESSION_TEXT = "Чтобы начать оценку ремонта, нажмите кнопку «Начать»."


# Nudge texts for the "user has an active session but typed/did something
# off-script" branch. Keyed by (mode, status) — we never want to tell a
# user "use the buttons in the last message" when the last message was a
# plain-text "send me a photo" prompt with no buttons attached, which is
# exactly what happened before this helper existed. See bug #2 in the
# accompanying UX fix round.
_NUDGE_MANUAL_WAITING_FOR_INPUT = (
    "Вы в режиме ручного ввода. Используйте кнопки в последнем сообщении, "
    "чтобы выбрать деталь и повреждение, или нажмите «Начать», чтобы "
    "открыть новую заявку."
)
_NUDGE_ML_AWAITING_PHOTO = (
    "Жду фотографию повреждения автомобиля. Пришлите её сообщением или нажмите «Начать», чтобы открыть новую заявку."
)
_NUDGE_ML_PROCESSING = (
    "Ваш снимок уже обрабатывается — я пришлю результат, как только "
    "модель закончит анализ. Если хотите начать заново, нажмите «Начать»."
)
_NUDGE_PRICING = (
    "Используйте кнопки в последнем сообщении, чтобы подтвердить или "
    "исправить результат, или нажмите «Начать», чтобы открыть новую заявку."
)


def active_session_nudge(mode: str | None, status: str | None) -> str:
    """Return the right off-script nudge for ``(mode, status)``.

    Values are passed as raw strings (the backend returns them as strings
    in the ``get_active_request`` response) and matched against
    ``RequestMode`` / ``RequestStatus`` values. Unknown / missing inputs
    degrade to the generic PRICING copy — it's the least wrong default
    because it's the only status where "use the buttons" is always true.
    """

    mode_v = (mode or "").lower()
    status_v = (status or "").lower()

    if status_v in {"pricing", "done", "failed"}:
        return _NUDGE_PRICING
    if mode_v == "manual":
        return _NUDGE_MANUAL_WAITING_FOR_INPUT
    if mode_v == "ml":
        if status_v == "created":
            return _NUDGE_ML_AWAITING_PHOTO
        if status_v in {"queued", "processing"}:
            return _NUDGE_ML_PROCESSING
    return _NUDGE_PRICING


def _welcome_text(previous_abandoned: bool) -> str:
    """Prepend the "old session closed" notice when applicable.

    Clicking "Начать" / typing ``/start`` is an explicit restart intent —
    whatever non-terminal session the chat had is closed *before* the
    welcome screen is shown. Telling the user that it happened (vs.
    silently wiping their progress) lets them notice if they pressed the
    button by accident and still have time to course-correct on the
    mode-selection screen.
    """
    if previous_abandoned:
        return f"{PREVIOUS_REQUEST_ABANDONED_NOTICE}\n\n{WELCOME_TEXT}"
    return WELCOME_TEXT


async def handle_start(message: Message, backend: BackendClient) -> None:
    # The text-command entry point ("/start" / "Начать" / "начать").
    # Goes through the same abandon-then-show-welcome path as the
    # inline-button variant to keep behaviour identical regardless of
    # how the user triggered the restart.
    abandoned = await abandon_active_session(backend, message.peer_id)
    await message.answer(
        message=_welcome_text(abandoned is not None),
        keyboard=mode_selection_keyboard(),
    )


async def handle_start_callback(
    event: MessageEvent,
    payload: dict[str, Any],  # noqa: ARG001 - uniform handler signature
    backend: BackendClient,
    api: API,
) -> None:
    """Same as ``/start`` but invoked from the inline "Начать" button.

    Semantics: pressing "Начать" is an explicit restart. Any non-terminal
    session for this chat is abandoned *before* we redraw the mode
    keyboard — otherwise clicking the button mid-flow would leave the
    old session alive and the next mode pick would ambiguously "continue
    or restart". The user is told what happened via a short notice
    prefixed to the welcome text.
    """

    abandoned = await abandon_active_session(backend, event.peer_id)
    await api.messages.send(
        peer_id=event.peer_id,
        message=_welcome_text(abandoned is not None),
        keyboard=mode_selection_keyboard(),
        random_id=0,
    )


async def send_no_active_session_reply(api: API, peer_id: int) -> None:
    """Ask the user to press "Начать" and attach the keyboard that lets them."""
    await api.messages.send(
        peer_id=peer_id,
        message=NO_ACTIVE_SESSION_TEXT,
        keyboard=start_keyboard(),
        random_id=0,
    )
