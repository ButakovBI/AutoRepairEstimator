from httpx import ASGITransport, AsyncClient
from pytest import mark

from auto_repair_estimator.backend.main import app as backend_app


@mark.anyio
async def test_backend_health() -> None:
    async with AsyncClient(transport=ASGITransport(app=backend_app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
