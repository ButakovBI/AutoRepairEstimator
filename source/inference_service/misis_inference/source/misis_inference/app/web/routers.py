
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, UploadFile, status

from misis_inference.app.web.prediction_service import PredictionService
from misis_inference.models.prediction_response import PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
prediction_service = PredictionService()


@router.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(file: UploadFile, scenario_id: UUID):
    try:
        return await prediction_service.predict(file, scenario_id=scenario_id)
    except Exception as e:
        logger.error("[Inference] Prediction of file %s failed: %s", file, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"[Inference] Error processing frame: {str(e)}. try again"
        )
