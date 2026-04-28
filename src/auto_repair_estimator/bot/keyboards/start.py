"""Single-button inline keyboard used for onboarding and reject replies.

We attach this keyboard to:

* The first message a user ever sends that isn't a photo.
* Any free-text / legacy-callback nudge when there's no active session.
* The idle "what should I do?" reply to off-script messages.

A single inline "Начать" button is preferable to VK's persistent reply
keyboard here: persistent keyboards conflict with the per-message inline
keyboards we already use throughout the flow, and force us to re-send the
reply keyboard on every message. A one-shot inline button disappears after
tap (good — the user is now in a scenario) and pairs cleanly with our
existing ``cmd``-dispatch convention.
"""

from __future__ import annotations

from vkbottle import Callback, Keyboard, KeyboardButtonColor


def start_keyboard() -> str:
    kb = Keyboard(inline=True)
    kb.add(Callback("Начать", payload={"cmd": "start"}), color=KeyboardButtonColor.POSITIVE)
    return str(kb.get_json())
