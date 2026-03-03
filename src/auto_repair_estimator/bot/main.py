from fastapi import FastAPI


app = FastAPI(title="Auto Repair Estimator Telegram Bot Service")


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("auto_repair_estimator.bot.main:app", host="0.0.0.0", port=8001, reload=True)

