from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Initialize Kafka consumers/producers and model resources here.
    yield
    # Shutdown resources here.


app = FastAPI(title="Auto Repair Estimator ML Worker", lifespan=lifespan)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("auto_repair_estimator.ml_worker.main:app", host="0.0.0.0", port=8002, reload=True)

