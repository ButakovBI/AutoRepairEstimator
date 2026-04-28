from vkbottle import Callback, Keyboard, KeyboardButtonColor


def mode_selection_keyboard() -> str:
    kb = Keyboard(inline=True)
    kb.add(Callback("С фотографией (ML)", payload={"cmd": "mode", "m": "ml"}), color=KeyboardButtonColor.PRIMARY)
    kb.row()
    kb.add(Callback("Ручной ввод", payload={"cmd": "mode", "m": "manual"}), color=KeyboardButtonColor.SECONDARY)
    return str(kb.get_json())
