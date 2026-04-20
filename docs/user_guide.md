# User Guide — Auto Repair Estimator Bot

This VK bot helps you estimate car repair costs by either uploading a photo of the damage or selecting damage types manually.

## Getting Started

Send `/start` to the bot **or** press the **«Начать»** button (the bot
surfaces it automatically whenever you don't have an active scenario).
You will be asked to choose a mode:

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
3. Select the **type of damage**. The list of types depends on the part:
   - **Body panels** (door, fenders, trunk, hood, roof, bumper) — Scratch / Dent / Paint chip / Rust / Crack.
   - **Glass** (windshields, side window) — Broken glass (the only failure mode the shop prices).
   - **Headlight** — Broken headlight (the only priced option; scratches on a headlight are not offered).
   - **Wheel** — Flat tire (routed to a tyre shop).
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

### How the bot aggregates damages into the estimate

The list of detected damages shown to the user is always the full list
(e.g. "Дверь — Царапина, Дверь — Царапина, Дверь — Трещина"), but the
**priced total** applies two business rules before summing:

1. **One occurrence per damage type per part.** Multiple damages of the
   same type on the same physical part are charged as one — painting a
   door covers every scratch on it, so 3 scratches on one door are
   priced as 1 scratch.
2. **Replacement supersedes other repairs on the same part.** If a part
   has any replacement-class damage (crack, paint chip, broken glass,
   broken headlight) alongside paint/dent damages, the part is priced
   purely as replacement — you don't paint a panel you're replacing.

Example matching the thesis spec:

| Detected (displayed) | Priced as |
|----------------------|-----------|
| Дверь — 2 царапины + 2 трещины | Дверь — замена (1 row) |
| Лобовое стекло — битое стекло | Стекло — замена |
| Фара — разбитая фара | Фара — замена |
| Переднее крыло — 2 вмятины + 3 царапины | Крыло — вмятина + Крыло — царапина (2 rows, no replacement) |

---

## Tips for Best Results

- Take photos in good lighting (natural daylight is best).
- Photograph from close range so the damaged area fills most of the frame.
- Keep the photo in focus.
- One part per photo works best — avoid photographing the whole car at once.

---

## Starting Over

To start a new estimation session, send `/start` at any time, or press
the **«Начать»** button that the bot attaches to onboarding / "out of
session" replies.

## What if I press an old button or write something random?

The bot's scenarios live on server-side state. If your previous session
has already finished (confirmed pricing, timed out) or a button you
tap is from a completed request, the bot replies with
*"Чтобы начать оценку ремонта, нажмите «Начать»"* and attaches the
button. Free-text messages outside an active session get the same
nudge — the only inputs that always start a new scenario are `/start`
and sending a photo.

---

## Deploying Trained ML Models (operator note)

The ML worker loads two YOLOv8-seg networks from `/app/models/` inside the
container, which `docker/docker-compose.yml` maps to the host path
`docker/models/`:

| File on host | Loaded as | Produces |
|--------------|-----------|----------|
| `docker/models/parts.pt`   | parts segmentation | `PartType` detections on the full image |
| `docker/models/damages.pt` | damage segmentation | `DamageType` detections on each cropped part |

Class names inside the `.pt` file **must** match the enum values in
`src/auto_repair_estimator/backend/domain/value_objects/request_enums.py`
(see `docker/models/README.md`). On load the worker cross-checks
`model.names` against those enums and logs warnings for any mismatch;
unknown classes are also skipped at inference time so a mis-aligned model
cannot poison downstream storage.

To roll out a freshly trained checkpoint:

```bash
cp /path/to/runs/damages/weights/best.pt docker/models/damages.pt
docker compose -f docker/docker-compose.yml restart ml_worker
docker logs -f auto-repair-ml-worker    # watch for "model loaded" line
```

If the worker starts with only one model present (e.g. `damages.pt` is
ready but `parts.pt` isn't trained yet), `ml_worker/main.py` skips loading
the missing one — the worker still starts, but inference requests requiring
that stage will fail fast with a structured error.
