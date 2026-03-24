from __future__ import annotations

from vkbottle.bot import Message

from auto_repair_estimator.bot.keyboards.mode_selection import mode_selection_keyboard


async def handle_start(message: Message) -> None:
    text = (
        "Добро пожаловать! Я помогу оценить стоимость ремонта вашего автомобиля.\n\n"
        "Для этого вы можете:\n"
        "- Загрузить фотографию повреждения — я проанализирую его с помощью AI\n"
        "- Указать повреждения вручную — выбрав деталь и тип повреждения\n\n"
        "Выберите режим работы:"
    )
    await message.answer(message=text, keyboard=mode_selection_keyboard())
