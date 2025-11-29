import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from uuid import UUID

import cv2

from misis_runner.inference_client import InferenceClient
from misis_runner.models.bounding_box import BoundingBox
from misis_runner.s3.s3_client import S3Client

FRAME_SKIP = 10
SIZE = (640, 480)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VideoProcessor:
    def __init__(
        self,
        scenario_id: UUID,
        s3_video_key: str,
        s3_client: S3Client,
        inference_client: InferenceClient,
        resume_from_frame: int = 0
    ):
        self.scenario_id = scenario_id
        self.s3_video_key = s3_video_key
        self.s3_client = s3_client
        self.inference_client = inference_client
        self._stop_event = asyncio.Event()
        self.is_stopping = False
        self.last_processed_frame = resume_from_frame
        self.processed_frames = 0
        self.temp_video_path = None
        self.current_frame_task = None

    async def process(self):
        if self.last_processed_frame == -1:
            logger.info("[Runner] This scenario already completed, check predictions")
            return
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            cap = None
            try:
                if self._stop_event.is_set():
                    return
                self.temp_video_path = await self.s3_client.download_video(self.s3_video_key)
                if self._stop_event.is_set():
                    return

                logger.info("[Runner] Video processor: starting process")

                cap = await loop.run_in_executor(executor, cv2.VideoCapture, self.temp_video_path)
                is_opened = await loop.run_in_executor(executor, cap.isOpened)
                if not is_opened:
                    raise ValueError(f"Failed to open video {self.temp_video_path}")

                if self.last_processed_frame > 0:
                    await loop.run_in_executor(
                        executor,
                        cap.set,
                        cv2.CAP_PROP_POS_FRAMES,
                        self.last_processed_frame
                    )
                    logger.info(f"[Runner] Resuming from frame {self.last_processed_frame}")

                while not self._stop_event.is_set():
                    ret, frame = await loop.run_in_executor(executor, cap.read)
                    self.last_processed_frame += 1

                    if not ret:
                        logger.info(f"[Runner] Reached end of video at frame {self.last_processed_frame}")
                        self.last_processed_frame = -1
                        break

                    if self.last_processed_frame % FRAME_SKIP != 0:
                        continue
                    try:
                        self.current_frame_task = asyncio.create_task(
                            self._process_frame(frame)
                        )
                        await self.current_frame_task
                    except asyncio.CancelledError:
                        logger.debug(f"[Runner] Frame processing cancelled for {self.scenario_id}")
                        break
                    except Exception as e:
                        logger.error(f"[Runner] Frame processing error: {str(e)}")
            except Exception as e:
                logger.error(f"[Runner] Video processing failed: {str(e)}")
            finally:
                if cap:
                    await loop.run_in_executor(executor, cap.release)
                if self.temp_video_path and os.path.exists(self.temp_video_path):
                    os.unlink(self.temp_video_path)
                self.is_stopping = True
                logger.info(f"[Runner] Completed. Scenario {self.scenario_id}. Processed {self.processed_frames} frames")

    def stop(self):
        self.is_stopping = True
        self._stop_event.set()
        if self.current_frame_task:
            self.current_frame_task.cancel()

    def _preprocess(self, frame):
        if frame is None:
            raise ValueError("Empty frame received")
        return cv2.resize(frame, SIZE) / 255.0

    async def _process_frame(self, frame):
        if frame is None:
            logger.warning("Empty frame skipped")
            return

        loop = asyncio.get_running_loop()
        _, img_bytes = await loop.run_in_executor(None, cv2.imencode, '.jpg', frame)
        processed_frame = img_bytes.tobytes()
        predictions = await self.inference_client.predict_frame(
            frame_bytes=processed_frame,
            scenario_id=self.scenario_id
        )
        await self._save_predictions(predictions)
        self.processed_frames += 1
        await asyncio.sleep(5)

    async def _save_predictions(self, predictions: list[BoundingBox]):
        try:
            await self.s3_client.save_predictions(
                scenario_id=self.scenario_id,
                frame_number=self.last_processed_frame,
                predictions=predictions
            )
            logger.info(f"[Runner] Saved predictions for frame {self.last_processed_frame}")
        except Exception as e:
            logger.error(f"[Runner] Failed to save predictions: {str(e)}")
