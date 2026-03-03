from httpx import AsyncClient
from pytest import mark

from auto_repair_estimator.backend.main import app as backend_app
from auto_repair_estimator.bot.main import app as bot_app
from auto_repair_estimator.ml_worker.main import app as ml_app


@mark.anyio
async def test_backend_health() -> None:
    async with AsyncClient(app=backend_app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@mark.anyio
async def test_bot_health() -> None:
    async with AsyncClient(app=bot_app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@mark.anyio
async def test_ml_worker_health() -> None:
    async with AsyncClient(app=ml_app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

