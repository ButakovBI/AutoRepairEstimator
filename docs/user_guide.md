# User Guide — Auto Repair Estimator Bot

This VK bot helps you estimate car repair costs by either uploading a photo of the damage or selecting damage types manually.

## Getting Started

Send `/start` to the bot. You will be asked to choose a mode:

| Mode | When to use |
|------|-------------|
| 📷 С фотографией (ML) | Upload a clear photo of the damage — the AI will detect parts and damage automatically |
| ✏️ Ручной ввод | Manually select the damaged part and damage type using inline buttons |

---

## ML Mode (Photo)

1. Choose **"С фотографией (ML)"**.
2. Send a **clear, well-lit photo** where the damaged area is clearly visible and the part is recognizable.
   - If you attach **several photos in one message**, the bot creates one inference request per photo and reports how many were accepted.
   - Supported formats: JPEG, PNG
   - Maximum size: 10 MB
   - Minimum resolution: 320×320 pixels
3. The bot will confirm receipt and start processing (~15 seconds).
4. When done, you will receive:
   - The original photo with damage segmentation masks overlaid
   - A list of detected damages
5. You can then:
   - **Confirm** if the results are correct
   - **Edit** to change a damage type, delete false positives, or add missed damages
6. After confirmation, you receive the estimated repair cost and duration.

---

## Manual Mode

1. Choose **"Ручной ввод"**.
2. Select a **damaged part** from the keyboard (e.g., Hood / Bumper front / Door).
3. Select the **type of damage** (Scratch / Dent / Crack / Rust / Paint chip).
4. The bot confirms the added damage and offers to add more or proceed.
5. Press **"Подтвердить"** to get the estimate.

---

## Pricing

Pricing reflects the auto repair shop's official rate card (thesis tables 5 & 6). Every damage is mapped to one of four work types and the corresponding cost/duration range is applied per part:

| Damage type → work type | How the shop prices it |
|-------------------------|------------------------|
| Scratch → painting | 10–18 тыс. руб., 1 day (polishing alternative: 1 h, 1 000 руб. — surfaced as a note) |
| Rust → painting | same as scratch |
| Dent → straightening + painting | 3–35 тыс. руб. and 1–3 days depending on the part |
| Paint chip / Crack → part replacement | varies per part (e.g. door 20 тыс. руб./1.5–2 дня, roof 75–100 тыс. руб./5 дней) |
| Broken glass → glass replacement | 3–10 тыс. руб., 1 day |
| Broken headlight → headlight replacement | 3 тыс. руб., 0.5 day |
| Flat tire / any wheel damage | Routed to a tyre shop — no body-shop estimate produced |

The bot renders both **per-damage ranges** (e.g. *Дверь — вмятина: 23 000–30 000 руб. (16–24 ч)*) and the **aggregate range** over all active damages. A dedicated note is appended at the bottom of the estimate whenever polishing is a cheaper scratch alternative, or when the user should visit a tyre shop.

---

## Tips for Best Results

- Take photos in good lighting (natural daylight is best).
- Photograph from close range so the damaged area fills most of the frame.
- Keep the photo in focus.
- One part per photo works best — avoid photographing the whole car at once.

---

## Starting Over

To start a new estimation session, send `/start` at any time.
