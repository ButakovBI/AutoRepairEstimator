import asyncio
import io
import logging
from uuid import UUID

from fastapi import UploadFile
from PIL import Image
from ultralytics import YOLO

from misis_inference.models.bounding_box import BoundingBox
from misis_inference.models.prediction_response import PredictionResponse


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PredictionService:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.model = YOLO('yolov8n.pt')
            logger.info("YOLO model initialized")
        return cls._instance

    async def predict(self, file: UploadFile, scenario_id: UUID) -> PredictionResponse:
        logger.info(f"[Inference] Do prediction for frame {file.filename}, scenario: {scenario_id}")
        bytes_data = await file.read()
        image = None

        try:
            loop = asyncio.get_running_loop()
            image = await loop.run_in_executor(
                None,
                lambda: Image.open(io.BytesIO(bytes_data)).convert("RGB")
            )

            results = await loop.run_in_executor(
                None,
                lambda: self.model.predict(source=image, verbose=False, imgsz=640)
            )

            predictions = []
            for res in results:
                for box in res.boxes:
                    predictions.append(BoundingBox(
                        x1=float(box.xyxy[0][0]),
                        y1=float(box.xyxy[0][1]),
                        x2=float(box.xyxy[0][2]),
                        y2=float(box.xyxy[0][3]),
                        class_name=res.names[int(box.cls[0])],
                        confidence=float(box.conf[0])
                    ))

            logger.info(f"[Inference] Prediction done for {file.filename}")
            return PredictionResponse(
                scenario_id=scenario_id,
                predictions=predictions
            )
        finally:
            if image:
                image.close()
