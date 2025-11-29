import json
import logging
from uuid import UUID

import aioboto3

from misis_scenario_api.models.bounding_box import BoundingBox
from misis_scenario_api.app.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class S3Client:
    def __init__(self):
        self.session = aioboto3.Session()

    async def upload_video(self, bucket: str,
                           s3_link: str, video_bytes: bytes) -> None:
        async with self.session.client(
            's3',
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY
        ) as client:
            await client.put_object(
                Bucket=bucket,
                Key=s3_link,
                Body=video_bytes
            )

    async def get_predictions(self, scenario_id: UUID) -> list[BoundingBox]:
        prefix = f"predictions/{scenario_id}/"
        predictions = []

        async with self.session.client(
            's3',
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY
        ) as client:
            list_objects = await client.list_objects_v2(
                Bucket=settings.S3_PREDICTIONS_BUCKET,
                Prefix=prefix
            )

            for obj in list_objects.get('Contents', []):
                try:
                    response = await client.get_object(
                        Bucket=settings.S3_PREDICTIONS_BUCKET,
                        Key=obj['Key']
                    )
                    data = json.loads(await response['Body'].read())
                    predictions.append(data)
                except Exception as e:
                    logger.error(f"[API S3] S3 client: Failed to load prediction {obj['Key']}: {str(e)}")

        return sorted(predictions, key=lambda x: x["frame_number"])
