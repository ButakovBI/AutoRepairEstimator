from __future__ import annotations

import asyncio
import io
from datetime import timedelta

from loguru import logger
from minio import Minio
from minio.error import S3Error


class MinioStorageGateway:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False) -> None:
        url = endpoint.removeprefix("http://").removeprefix("https://")
        self._client = Minio(url, access_key=access_key, secret_key=secret_key, secure=secure)

    async def generate_presigned_put_url(self, key: str, expires: int = 3600) -> str:
        bucket, obj = self._split_key(key)
        url = await asyncio.to_thread(
            self._client.presigned_put_object,
            bucket,
            obj,
            expires=timedelta(seconds=expires),
        )
        logger.debug("Generated presigned PUT URL for key={}", key)
        return str(url)

    async def generate_presigned_get_url(self, key: str, expires: int = 3600) -> str:
        bucket, obj = self._split_key(key)
        url = await asyncio.to_thread(
            self._client.presigned_get_object,
            bucket,
            obj,
            expires=timedelta(seconds=expires),
        )
        return str(url)

    async def download(self, key: str) -> bytes:
        bucket, obj = self._split_key(key)
        response = await asyncio.to_thread(self._client.get_object, bucket, obj)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        logger.debug("Downloaded {} bytes from key={}", len(data), key)
        return data

    async def object_exists(self, key: str) -> bool:
        bucket, obj = self._split_key(key)
        try:
            await asyncio.to_thread(self._client.stat_object, bucket, obj)
            return True
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return False
            raise

    async def ensure_buckets(self, *bucket_names: str) -> None:
        for bucket in bucket_names:
            exists = await asyncio.to_thread(self._client.bucket_exists, bucket)
            if not exists:
                await asyncio.to_thread(self._client.make_bucket, bucket)
                logger.info("Created MinIO bucket={}", bucket)

    async def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
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
