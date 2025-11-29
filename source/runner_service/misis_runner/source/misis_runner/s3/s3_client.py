import asyncio
import json
import logging
import tempfile
from uuid import UUID

import aioboto3

from misis_runner.app.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class S3Client:
    def __init__(self):
        self.session = aioboto3.Session()
        self.retry_count = 3

    async def download_video(self, s3_key: str) -> str:
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            async with self.session.client(
                's3',
                endpoint_url=settings.S3_ENDPOINT,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY
            ) as client:
                await client.download_file(
                    Bucket=settings.S3_VIDEOS_BUCKET,
                    Key=s3_key,
                    Filename=temp_path,
                )
                logger.info(f"[Runner] S3 client: Video downloaded to {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"[Runner] S3 client: Failed to download video {s3_key}: {str(e)}")
            raise
        finally:
            if temp_file:
                temp_file.close()

    async def save_predictions(self, scenario_id: UUID, frame_number: int, predictions: list):
        key = f"predictions/{scenario_id}/{frame_number}.json"
        data = {
            "scenario_id": str(scenario_id),
            "frame_number": frame_number,
            "predictions": [box.dict() for box in predictions]
        }
        for attempt in range(self.retry_count):
            try:
                async with self.session.client(
                    's3',
                    endpoint_url=settings.S3_ENDPOINT,
                    aws_access_key_id=settings.S3_ACCESS_KEY,
                    aws_secret_access_key=settings.S3_SECRET_KEY
                ) as client:
                    await client.put_object(
                        Bucket=settings.S3_PREDICTIONS_BUCKET,
                        Key=key,
                        Body=json.dumps(data).encode('utf-8'),
                        ContentType='application/json',
                    )
                    logger.info(f"[Runner] S3 client: Predictions saved to S3: {key}")
                return
            except Exception as e:
                logger.error(f"Failed to save predictions to S3: {str(e)}, retrying... ({attempt + 1}/{self.retry_count})")
                if attempt == self.retry_count - 1:
                    raise
                await asyncio.sleep(1)
