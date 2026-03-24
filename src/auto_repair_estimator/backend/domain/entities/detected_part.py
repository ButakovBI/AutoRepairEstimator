from dataclasses import dataclass

from auto_repair_estimator.backend.domain.value_objects.request_enums import PartType


@dataclass
class DetectedPart:
    id: str
    request_id: str
    part_type: PartType
    confidence: float
    x: float
    y: float
    width: float
    height: float
    mask_image_key: str | None = None
    crop_image_key: str | None = None
