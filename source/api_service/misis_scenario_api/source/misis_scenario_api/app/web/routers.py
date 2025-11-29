import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from misis_scenario_api.app.web.scenario_service import ScenarioService
from misis_scenario_api.database.database import get_db_session
from misis_scenario_api.kafka.producer import Producer
from misis_scenario_api.models.constants.command_type import CommandType
from misis_scenario_api.models.prediction_response import PredictionResponse
from misis_scenario_api.models.scenario_status_response import \
    ScenarioStatusResponse
from misis_scenario_api.s3.s3_client import S3Client

router = APIRouter()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@router.post(
    "/scenario/",
    response_model=ScenarioStatusResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def create_scenario(
    video: UploadFile,
    db: AsyncSession = Depends(get_db_session),
    producer: Producer = Depends(),
    s3_client: S3Client = Depends()
):
    try:
        service = ScenarioService(db=db, producer=producer, s3_client=s3_client)
        res = await service.create_scenario(video=video)
        return res
    except Exception as e:
        logger.error(f"[API] Creation scenario failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Creation scenario failed: {str(e)}. Try to restart scenario"
        )


@router.post(
    "/scenario/{scenario_id}/",
    response_model=ScenarioStatusResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def update_scenario(
    scenario_id: UUID,
    command: CommandType,
    db: AsyncSession = Depends(get_db_session),
    producer: Producer = Depends()
):
    try:
        service = ScenarioService(db=db, producer=producer, s3_client=None)
        return await service.update_scenario(scenario_id=scenario_id, command=command.value)
    except Exception as e:
        logger.error(f"[API] Updating scenario failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Updating scenario failed: {str(e)}"
        )


@router.get(
    "/scenario/{scenario_id}/",
    response_model=ScenarioStatusResponse,
    status_code=status.HTTP_200_OK
)
async def get_scenario_status(
    scenario_id: UUID,
    db: AsyncSession = Depends(get_db_session)
):
    try:
        service = ScenarioService(db=db, producer=None, s3_client=None)
        return await service.get_scenario_status(scenario_id=scenario_id)
    except Exception as e:
        logger.error(f"[API] Getting scenario status failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scenario status: {str(e)}"
        )


@router.get(
    "/prediction/{scenario_id}/",
    response_model=list[PredictionResponse],
)
async def get_predictions(
    scenario_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    s3_client: S3Client = Depends(),
):
    try:
        service = ScenarioService(db=db, producer=None, s3_client=s3_client)
        return await service.get_predictions(scenario_id=scenario_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get predictions: {str(e)}"
        )
