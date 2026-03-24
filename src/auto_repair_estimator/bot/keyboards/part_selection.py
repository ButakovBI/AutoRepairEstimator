from vkbottle import Callback, Keyboard

from auto_repair_estimator.bot.labels import PART_LABELS

# VK messages.send inline keyboard: error 911 if too many buttons or rows.
_VK_INLINE_MAX_BUTTONS = 10
_VK_PARTS_PER_ROW = 2


def _keyboard_for_part_items(request_id: str, items: list[tuple[str, str]]) -> str:
    if not items:
        raise ValueError("items must be non-empty")
    if len(items) > _VK_INLINE_MAX_BUTTONS:
        msg = f"chunk exceeds VK inline button limit ({_VK_INLINE_MAX_BUTTONS})"
        raise ValueError(msg)
    kb = Keyboard(inline=True)
    for i, (part_type, label) in enumerate(items):
        kb.add(Callback(label, payload={"cmd": "part", "rid": request_id, "pt": part_type}))
        if (i + 1) % _VK_PARTS_PER_ROW == 0 and i < len(items) - 1:
            kb.row()
    return kb.get_json()


def part_selection_keyboards_list(request_id: str) -> list[str]:
    """Return one JSON keyboard per VK message; each has at most 10 part buttons."""
    items = list(PART_LABELS.items())
    keyboards: list[str] = []
    for offset in range(0, len(items), _VK_INLINE_MAX_BUTTONS):
        chunk = items[offset : offset + _VK_INLINE_MAX_BUTTONS]
        keyboards.append(_keyboard_for_part_items(request_id, chunk))
    return keyboards
