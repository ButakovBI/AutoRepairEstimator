from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from auto_repair_estimator.backend.domain.entities.repair_request import RepairRequest
from auto_repair_estimator.backend.domain.value_objects.request_enums import RequestStatus


class InvalidStatusTransitionError(ValueError):
    def __init__(self, from_status: RequestStatus, to_status: RequestStatus) -> None:
        message = f"invalid transition {from_status.value} -> {to_status.value}"
        super().__init__(message)
        self.from_status = from_status
        self.to_status = to_status


@dataclass(frozen=True)
class RequestStateMachine:
    def can_transition(self, from_status: RequestStatus, to_status: RequestStatus) -> bool:
        if from_status is RequestStatus.DONE or from_status is RequestStatus.FAILED:
            return False
        if from_status is to_status:
            return True
        if from_status is RequestStatus.CREATED and to_status in {RequestStatus.QUEUED, RequestStatus.PRICING, RequestStatus.FAILED}:
            return True
        if from_status is RequestStatus.QUEUED and to_status in {RequestStatus.PROCESSING, RequestStatus.FAILED}:
            return True
        if from_status is RequestStatus.PROCESSING and to_status in {RequestStatus.PRICING, RequestStatus.FAILED}:
            return True
        if from_status is RequestStatus.PRICING and to_status in {RequestStatus.DONE, RequestStatus.FAILED}:
            return True
        return False

    def transition(self, request: RepairRequest, to_status: RequestStatus) -> RepairRequest:
        if not self.can_transition(request.status, to_status):
            raise InvalidStatusTransitionError(request.status, to_status)
        if request.status is to_status:
            return request
        updated_at = datetime.now(timezone.utc)
        if updated_at <= request.updated_at:
            updated_at = request.updated_at + timedelta(microseconds=1)
        return request.with_status(to_status, updated_at=updated_at)

