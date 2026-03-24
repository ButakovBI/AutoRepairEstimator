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

Pricing is based on a fixed lookup table for the auto repair shop:

| Damage Type | Approximate cost (RUB) | Approximate time |
|-------------|----------------------|-----------------|
| Scratch | 400–1 200 | 0.5–1.5 h |
| Dent | 1 500–3 500 | 2–3.5 h |
| Crack | 2 500–5 000 | 3–5 h |
| Rust | 2 000–3 500 | 2.5–4 h |
| Paint chip | 400–600 | 0.5 h |

Final estimate depends on the specific part and damage combination.

---

## Tips for Best Results

- Take photos in good lighting (natural daylight is best).
- Photograph from close range so the damaged area fills most of the frame.
- Keep the photo in focus.
- One part per photo works best — avoid photographing the whole car at once.

---

## Starting Over

To start a new estimation session, send `/start` at any time.
