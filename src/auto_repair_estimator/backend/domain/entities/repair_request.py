from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    RequestMode,
    RequestStatus,
)


@dataclass
class RepairRequest:
    id: str
    chat_id: int
    user_id: int | None
    mode: RequestMode
    status: RequestStatus
    created_at: datetime
    updated_at: datetime
    timeout_at: datetime
    original_image_key: str | None = None
    composited_image_key: str | None = None
    ml_error_code: str | None = None
    ml_error_message: str | None = None

    @classmethod
    def new(cls, request_id: str, chat_id: int, user_id: int | None, mode: RequestMode) -> "RepairRequest":
        now = datetime.now(UTC)
        timeout = now + timedelta(minutes=5)
        status = RequestStatus.PRICING if mode is RequestMode.MANUAL else RequestStatus.CREATED
        return cls(
            id=request_id,
            chat_id=chat_id,
            user_id=user_id,
            mode=mode,
            status=status,
            created_at=now,
            updated_at=now,
            timeout_at=timeout,
        )

    def with_status(self, status: RequestStatus, updated_at: datetime | None = None) -> "RepairRequest":
        new_updated_at = updated_at or datetime.now(UTC)
        return RepairRequest(
            id=self.id,
            chat_id=self.chat_id,
            user_id=self.user_id,
            mode=self.mode,
            status=status,
            created_at=self.created_at,
            updated_at=new_updated_at,
            timeout_at=self.timeout_at,
            original_image_key=self.original_image_key,
            composited_image_key=self.composited_image_key,
            ml_error_code=self.ml_error_code,
            ml_error_message=self.ml_error_message,
        )
