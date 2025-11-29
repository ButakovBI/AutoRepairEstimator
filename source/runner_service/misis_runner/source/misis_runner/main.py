import asyncio
import logging

from misis_runner.runner import Runner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def main():
    runner = Runner()
    shutdown_event = asyncio.Event()
    try:
        await runner.start()
        await shutdown_event.wait()
    except KeyboardInterrupt:
        logger.info("[Runner] Received exit signal")
    finally:
        await runner.stop()
        logger.info("[Runner] Runner shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
