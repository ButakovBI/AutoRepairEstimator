# Auto Repair Estimator

A VK bot for estimating car repair costs using computer vision (YOLOv8-seg) or manual damage selection.

## Features

- **ML Mode**: Upload a photo of the damaged car — the system automatically detects parts and damage using two YOLOv8-seg models, overlays segmentation masks, and returns repair cost and time estimates.
- **Manual Mode**: Select damaged parts and damage types via inline keyboards — get instant pricing without photo analysis.
- **Damage Editing**: After ML inference, users can confirm, edit, or add/delete detected damages before pricing.
- **Reliable Async Pipeline**: Kafka-based message queue with transactional outbox pattern ensures at-least-once delivery.
- **Heartbeat Watchdog**: Automatically marks timed-out requests as failed and notifies users.

## Architecture

The system follows Clean Architecture with 3 services:

| Service | Technology | Port |
|---------|-----------|------|
| Backend API | FastAPI + asyncpg | 8000 |
| Bot | vkbottle 4.x | — |
| ML Worker | YOLOv8-seg + Kafka | — |

**Infrastructure**: PostgreSQL 16, Apache Kafka, MinIO S3.

> **Why no Nginx?** The bot uses the VK Long Poll API (`vkbottle.run_polling`):
> it initiates outbound HTTPS connections to VK and does not need to expose an
> HTTP endpoint to receive webhooks, so a reverse proxy in front of the bot is
> unnecessary. The backend is reached directly by internal services over the
> Docker network.

See the `.cursor/plans/` directory for detailed C4 diagrams and implementation plan.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- `.env` file (copy from `.env.example`)

```bash
cp .env.example .env
# Edit .env — set VK_GROUP_TOKEN (and VK_GROUP_ID if you use it)
docker compose -f docker/docker-compose.yml up -d
```

The backend API will be available at `http://localhost:8000`.

### Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## Project Structure

```
src/auto_repair_estimator/
  backend/          # FastAPI + domain + use cases + adapters
  bot/              # vkbottle bot + handlers + keyboards
  ml_worker/        # YOLOv8-seg inference pipeline
ml/                 # Training scripts
docker/             # Dockerfiles + docker-compose + init.sql
tests/              # Unit + integration tests
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/requests` | Create repair request (ML or manual mode) |
| GET | `/v1/requests/{id}` | Get request with detected damages |
| POST | `/v1/requests/{id}/photo` | Confirm photo upload (transitions CREATED→QUEUED) |
| POST | `/v1/requests/{id}/damages` | Add damage manually |
| PATCH | `/v1/requests/{id}/damages/{damage_id}` | Edit damage type |
| DELETE | `/v1/requests/{id}/damages/{damage_id}` | Soft-delete damage |
| POST | `/v1/requests/{id}/confirm` | Calculate pricing and confirm (→DONE) |
| GET | `/health` | Health check |

## Request Lifecycle

```
CREATED → QUEUED → PROCESSING → PRICING → DONE
    ↓         ↓          ↓          ↓
   FAILED   FAILED     FAILED     FAILED  (heartbeat timeout)
```

Manual mode starts directly in PRICING state.

## ML Pipeline

1. **Parts Detection** (`yolov8m-seg`): 12 car parts (door, front/rear fender, trunk, hood, roof, headlight, front/rear windshield, side window, wheel, bumper), confidence ≥ 0.7.
2. **Cropping**: crops every detected part; excluded parts are configurable via `MLWorkerConfig.crop_excluded_parts`.
3. **Damage Detection** (`yolov8m-seg`): 8 damage types per crop (scratch, dent, paint_chip, rust, crack, broken_glass, flat_tire, broken_headlight).
4. **Composition**: alpha-blend masks onto original image.
5. **Result**: publish to Kafka `inference_results` topic.

### Training Models

```bash
python ml/train_parts.py --data data/parts.yaml --epochs 100
python ml/train_damages.py --data data/damages.yaml --epochs 100
python ml/evaluate.py --model runs/parts/parts_seg_v1/weights/best.pt --data data/parts.yaml
python ml/export_model.py --model runs/parts/.../best.pt --output docker/models/parts.pt
```

## Testing

```bash
# All tests with coverage
pytest tests/ --cov=auto_repair_estimator --cov-fail-under=70

# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Type check
mypy src/
```

## CI/CD

GitHub Actions runs on every push/PR:
1. `ruff check` — linting
2. `ruff format --check` — formatting
3. `mypy` — type checking
4. `pytest --cov-fail-under=70` — tests with coverage gate
