from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from auto_repair_estimator.backend.domain.value_objects.request_enums import (
    RequestMode,
    RequestStatus,
)


@dataclass
class RepairRequest:
    """Aggregate root for a single user session.

    ``idempotency_key`` is the composite ``f"{chat_id}:{message_id}"`` that
    VK attaches to each photo message. Persisting it with a UNIQUE
    constraint (see ``docker/init.sql``) lets the backend deduplicate
    retransmitted VK events at the database level — a second ``INSERT``
    with the same key simply fails, and the use case returns the existing
    request instead of creating a new one.
    """

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
    idempotency_key: str | None = None

    @classmethod
    def new(
        cls,
        request_id: str,
        chat_id: int,
        user_id: int | None,
        mode: RequestMode,
        idempotency_key: str | None = None,
    ) -> "RepairRequest":
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
            idempotency_key=idempotency_key,
        )

    def with_status(
        self,
        status: RequestStatus,
        updated_at: datetime | None = None,
    ) -> "RepairRequest":
        new_updated_at = updated_at or datetime.now(UTC)
        # A state transition is, by definition, activity on this request;
        # the HeartbeatWatchdog should therefore treat it as a heartbeat
        # and push the timeout deadline forward. Terminal transitions don't
        # need this (the watchdog ignores DONE/FAILED anyway), but it's
        # harmless to refresh in those cases too.
        new_timeout_at = new_updated_at + timedelta(minutes=5)
        return RepairRequest(
            id=self.id,
            chat_id=self.chat_id,
            user_id=self.user_id,
            mode=self.mode,
            status=status,
            created_at=self.created_at,
            updated_at=new_updated_at,
            timeout_at=new_timeout_at,
            original_image_key=self.original_image_key,
            composited_image_key=self.composited_image_key,
            ml_error_code=self.ml_error_code,
            ml_error_message=self.ml_error_message,
            idempotency_key=self.idempotency_key,
        )

    def with_extended_timeout(self, new_timeout_at: datetime) -> "RepairRequest":
        """Return a copy with ``timeout_at`` pushed further into the future.

        Active user interactions (adding/editing damages, uploading a photo)
        must refresh this so the HeartbeatWatchdog doesn't kill a session
        where the user is clearly still engaged.
        """
        return RepairRequest(
            id=self.id,
            chat_id=self.chat_id,
            user_id=self.user_id,
            mode=self.mode,
            status=self.status,
            created_at=self.created_at,
            updated_at=datetime.now(UTC),
            timeout_at=new_timeout_at,
            original_image_key=self.original_image_key,
            composited_image_key=self.composited_image_key,
            ml_error_code=self.ml_error_code,
            ml_error_message=self.ml_error_message,
            idempotency_key=self.idempotency_key,
        )
