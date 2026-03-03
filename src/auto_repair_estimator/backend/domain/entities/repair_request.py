from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    RequestMode,
    RequestStatus,
)


@dataclass
class RepairRequest:
    id: str
    telegram_chat_id: int
    telegram_user_id: Optional[int]
    mode: RequestMode
    status: RequestStatus
    created_at: datetime
    updated_at: datetime
    timeout_at: datetime
    original_image_key: Optional[str] = None
    composited_image_key: Optional[str] = None
    ml_error_code: Optional[str] = None
    ml_error_message: Optional[str] = None

    @classmethod
    def new(cls, request_id: str, chat_id: int, user_id: Optional[int], mode: RequestMode) -> "RepairRequest":
        now = datetime.now(timezone.utc)
        timeout = now + timedelta(minutes=5)
        status = RequestStatus.PRICING if mode is RequestMode.MANUAL else RequestStatus.CREATED
        return cls(
            id=request_id,
            telegram_chat_id=chat_id,
            telegram_user_id=user_id,
            mode=mode,
            status=status,
            created_at=now,
            updated_at=now,
            timeout_at=timeout,
        )

    def with_status(self, status: RequestStatus, updated_at: Optional[datetime] = None) -> "RepairRequest":
        new_updated_at = updated_at or datetime.now(timezone.utc)
        return RepairRequest(
            id=self.id,
            telegram_chat_id=self.telegram_chat_id,
            telegram_user_id=self.telegram_user_id,
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

