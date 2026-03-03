from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from auto_repair_estimator.backend.api import router as requests_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


app = FastAPI(title="Auto Repair Estimator Backend", lifespan=lifespan)
app.include_router(requests_router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("auto_repair_estimator.backend.main:app", host="0.0.0.0", port=8000, reload=True)

