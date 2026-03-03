from dataclasses import dataclass
from typing import Optional

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageSource, DamageType


@dataclass
class DetectedDamage:
    id: str
    request_id: str
    damage_type: DamageType
    source: DamageSource
    is_deleted: bool
    part_id: Optional[str] = None
    confidence: Optional[float] = None
    mask_image_key: Optional[str] = None

