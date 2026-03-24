from dataclasses import dataclass

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageSource, DamageType, PartType


@dataclass
class DetectedDamage:
    id: str
    request_id: str
    damage_type: DamageType
    part_type: PartType
    source: DamageSource
    is_deleted: bool
    part_id: str | None = None
    confidence: float | None = None
    mask_image_key: str | None = None
