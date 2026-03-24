from __future__ import annotations

import asyncio
import io

from loguru import logger
from minio import Minio


class S3Client:
    def __init__(self, endpoint: str, access_key: str, secret_key: str) -> None:
        url = endpoint.removeprefix("http://").removeprefix("https://")
        self._client = Minio(url, access_key=access_key, secret_key=secret_key, secure=False)

    async def download_image(self, key: str) -> bytes:
        bucket, obj = self._split_key(key)
        response = await asyncio.to_thread(self._client.get_object, bucket, obj)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        logger.debug("Downloaded {} bytes from key={}", len(data), key)
        return data  # type: ignore[return-value]

    async def upload_image(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        bucket, obj = self._split_key(key)
        await asyncio.to_thread(
            self._client.put_object,
            bucket,
            obj,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        logger.debug("Uploaded {} bytes to key={}", len(data), key)

    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        parts = key.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid key format (expected 'bucket/object'): {key}")
        return parts[0], parts[1]
