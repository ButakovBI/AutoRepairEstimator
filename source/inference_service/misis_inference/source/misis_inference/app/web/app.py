from fastapi import FastAPI

from misis_inference.app.web.routers import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="MISIS Inference service",
    )
    app.include_router(router)
    return app


app = create_app()
