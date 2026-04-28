from vkbottle import Callback, Keyboard

from auto_repair_estimator.bot.labels import PART_LABELS
from auto_repair_estimator.bot.vk_limits import VK_INLINE_DEFAULT_BUTTONS_PER_ROW, VK_INLINE_MAX_BUTTONS


def _keyboard_for_part_items(request_id: str, items: list[tuple[str, str]]) -> str:
    if not items:
        raise ValueError("items must be non-empty")
    if len(items) > VK_INLINE_MAX_BUTTONS:
        msg = f"chunk exceeds VK inline button limit ({VK_INLINE_MAX_BUTTONS})"
        raise ValueError(msg)
    kb = Keyboard(inline=True)
    for i, (part_type, label) in enumerate(items):
        kb.add(Callback(label, payload={"cmd": "part", "rid": request_id, "pt": part_type}))
        if (i + 1) % VK_INLINE_DEFAULT_BUTTONS_PER_ROW == 0 and i < len(items) - 1:
            kb.row()
    return kb.get_json()


def part_selection_keyboards_list(request_id: str) -> list[str]:
    """Return one JSON keyboard per VK message; each stays within VK inline limits."""
    items = list(PART_LABELS.items())
    keyboards: list[str] = []
    for offset in range(0, len(items), VK_INLINE_MAX_BUTTONS):
        chunk = items[offset: offset + VK_INLINE_MAX_BUTTONS]
        keyboards.append(_keyboard_for_part_items(request_id, chunk))
    return keyboards
