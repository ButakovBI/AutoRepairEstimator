import logging
from uuid import UUID

import httpx

from misis_runner.models.bounding_box import BoundingBox

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InferenceClient:
    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(base_url=base_url)

    async def predict_frame(self, frame_bytes, scenario_id: UUID) -> list[BoundingBox]:
        try:
            files = {'file': ('frame.jpg', frame_bytes)}
            response = await self.client.post(
                "/predict",
                files=files,
                params={"scenario_id": str(scenario_id)}
            )
            response.raise_for_status()
            data = response.json()

            if not isinstance(data.get("predictions"), list):
                raise ValueError("Invalid predictions format")
            logger.info("[Runner] Got predictions from inference service")
            return [BoundingBox(**box) for box in data["predictions"]]
        except Exception as e:
            logger.error(f"[Runner] Inference request failed: {str(e)}")
            raise

    async def close(self):
        await self.client.aclose()
