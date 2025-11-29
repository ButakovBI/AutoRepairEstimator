import logging
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import insert, select, update
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from misis_scenario_api.app.config import settings
from misis_scenario_api.database.tables.outbox import Outbox
from misis_scenario_api.database.tables.scenario import Scenario
from misis_scenario_api.kafka.producer import Producer
from misis_scenario_api.models.bounding_box import BoundingBox
from misis_scenario_api.models.constants.command_type import CommandType
from misis_scenario_api.models.constants.scenario_status import ScenarioStatus
from misis_scenario_api.models.prediction_response import PredictionResponse
from misis_scenario_api.models.scenario_status_response import \
    ScenarioStatusResponse
from misis_scenario_api.s3.s3_client import S3Client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ScenarioService:
    def __init__(self, db: AsyncSession | None, producer: Producer | None, s3_client: S3Client | None):
        self.db = db
        self.producer = producer
        self.s3_client = s3_client

    async def create_scenario(self, video: UploadFile) -> ScenarioStatusResponse:
        logger.info(f"[API Create] Create scenario for video {video.filename}")
        scenario_id = uuid4()
        s3_video_path = f"scenarios/{scenario_id}/{video.filename}"
        await self.s3_client.upload_video(
            bucket=settings.S3_VIDEOS_BUCKET,
            s3_link=s3_video_path,
            video_bytes=await video.read(),
        )
        logger.info(f"[API Create] Video uploaded to s3: {video.filename}")

        async with self.db.begin() as transaction:
            logger.info(f"[API Create] Inserting scenario {scenario_id}")
            await self.db.execute(
                insert(Scenario).values(
                    id=scenario_id,
                    status=ScenarioStatus.INIT_STARTUP.value,
                    video_path=s3_video_path
                )
            )
            logger.info(f"[API Create] Inserting scenario {scenario_id} to outbox")

            await self.db.execute(
                insert(Outbox).values(
                    id=uuid4(),
                    scenario_id=scenario_id,
                    event_type=ScenarioStatus.INIT_STARTUP.value,
                    payload={
                        "video_path": s3_video_path,
                        "resume_from_frame": 0,
                    }
                )
            )

            await transaction.commit()
            logger.info(f"[API Create] Transaction committed for {scenario_id}")

        return ScenarioStatusResponse(
            scenario_id=scenario_id,
            status=ScenarioStatus.INIT_STARTUP.value
        )

    async def update_scenario(self, scenario_id: UUID, command: str) -> ScenarioStatusResponse:
        logger.info(f"[API Update] Update scenario {scenario_id} with command {command}")

        try:
            async with self.db.begin() as transaction:
                scenario = await self.db.execute(
                    select(Scenario).where(Scenario.id == scenario_id)
                )
                scenario = scenario.scalar_one_or_none()

                if not scenario:
                    logger.error(f"[API Update] Scenario {scenario_id} not found")
                    raise NoResultFound(f"Scenario {scenario_id} not found")

                current_status = scenario.status
                new_status = None
                event_type = None
                payload = {}

                if command == CommandType.START.value:
                    if current_status in [
                        ScenarioStatus.INIT_STARTUP.value,
                        ScenarioStatus.IN_STARTUP_PROCESSING.value,
                        ScenarioStatus.ACTIVE.value
                    ]:
                        logger.info(f"[API Update] Scenario {scenario_id} already started or starting")
                        return ScenarioStatusResponse(
                            scenario_id=scenario_id,
                            status=current_status
                        )
                    elif current_status in [
                        ScenarioStatus.IN_SHUTDOWN_PROCESSING.value,
                        ScenarioStatus.INIT_SHUTDOWN.value
                    ]:
                        raise ValueError("Cannot start scenario while shutting down")
                    elif current_status in [ScenarioStatus.INACTIVE.value]:
                        new_status = ScenarioStatus.INIT_STARTUP.value
                        event_type = ScenarioStatus.INIT_STARTUP.value
                        payload = {"video_path": scenario.video_path}

                elif command == CommandType.STOP.value:
                    if current_status in [
                        ScenarioStatus.INIT_SHUTDOWN.value,
                        ScenarioStatus.IN_SHUTDOWN_PROCESSING.value,
                        ScenarioStatus.INACTIVE.value
                    ]:
                        logger.info(f"[API Update] Scenario {scenario_id} already stopped or stopping")
                        return ScenarioStatusResponse(
                            scenario_id=scenario_id,
                            status=current_status
                        )
                    elif current_status in [
                        ScenarioStatus.INIT_STARTUP.value,
                        ScenarioStatus.IN_STARTUP_PROCESSING.value,
                        ScenarioStatus.ACTIVE.value
                    ]:
                        new_status = ScenarioStatus.INIT_SHUTDOWN.value
                        event_type = ScenarioStatus.INIT_SHUTDOWN.value

                if new_status is None:
                    raise ValueError(f"Invalid command '{command}' for current status '{current_status}'")

                await self.db.execute(
                    update(Scenario)
                    .where(Scenario.id == scenario_id)
                    .values(status=new_status)
                )
                logger.info(f"[API Update] Scenario status changed to {new_status}")

                await self.db.execute(
                    insert(Outbox).values(
                        id=uuid4(),
                        scenario_id=scenario_id,
                        event_type=event_type,
                        payload=payload,
                    )
                )
                logger.info(f"[API Update] Insert to outbox '{event_type}'")

                await transaction.commit()
                logger.info("[API Update] Scenario status update success")

            return ScenarioStatusResponse(
                scenario_id=scenario_id,
                status=new_status
            )

        except NoResultFound as e:
            logger.warning(f"[API Update] Scenario not found: {scenario_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        except ValueError as e:
            logger.warning(f"[API Update] Invalid operation: {str(e)}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    async def get_scenario_status(self, scenario_id: UUID) -> ScenarioStatusResponse:
        logger.info(f"[API Status] Getting status for scenario {scenario_id}")

        try:
            result = await self.db.execute(
                select(Scenario)
                .where(Scenario.id == scenario_id)
            )
            scenario = result.scalar_one_or_none()

            if not scenario:
                logger.warning(f"[API Status] Scenario not found: {scenario_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Scenario not found"
                )

            logger.debug(f"[API Status] got status for '{scenario_id}': '{scenario.status}'")
            return ScenarioStatusResponse(
                scenario_id=scenario_id,
                status=scenario.status
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[API Status] Failed to get status for '{scenario_id}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get scenario status: {str(e)}"
            )

    async def get_predictions(self, scenario_id: UUID) -> list[PredictionResponse]:
        logger.info(f"[API Get Prediction] Getting predictions for scenario '{scenario_id}'")

        try:
            scenario_exists = await self.db.execute(
                select(Scenario.id)
                .where(Scenario.id == scenario_id)
            )
            if not scenario_exists.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Scenario not found"
                )

            predictions = await self.s3_client.get_predictions(scenario_id=scenario_id)

            logger.debug(f"[API Get Prediction] Found {len(predictions)} predictions for '{scenario_id}'")
            return [
                PredictionResponse(
                    scenario_id=scenario_id,
                    predictions=[
                        BoundingBox(**box) for box in pred["predictions"]
                    ]
                )
                for pred in predictions
            ]

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[API Get Prediction] Failed to get predictions for '{scenario_id}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get predictions: {str(e)}"
            )
