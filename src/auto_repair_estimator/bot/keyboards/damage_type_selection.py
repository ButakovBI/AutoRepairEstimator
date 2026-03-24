from vkbottle import Callback, Keyboard

from auto_repair_estimator.bot.labels import DAMAGE_LABELS


def damage_type_selection_keyboard(request_id: str, part_type: str) -> str:
    kb = Keyboard(inline=True)
    items = list(DAMAGE_LABELS.items())
    for i, (damage_type, label) in enumerate(items):
        kb.add(Callback(label, payload={"cmd": "dmg", "rid": request_id, "pt": part_type, "dt": damage_type}))
        if i % 2 == 1 and i < len(items) - 1:
            kb.row()
    return kb.get_json()
