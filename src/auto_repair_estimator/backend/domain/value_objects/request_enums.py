from enum import Enum


class RequestMode(str, Enum):
    ML = "ml"
    MANUAL = "manual"


class RequestStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    PROCESSING = "processing"
    PRICING = "pricing"
    DONE = "done"
    FAILED = "failed"


class PartType(str, Enum):
    BUMPER_FRONT = "bumper_front"
    BUMPER_REAR = "bumper_rear"
    DOOR_FRONT_LEFT = "door_front_left"
    DOOR_FRONT_RIGHT = "door_front_right"
    DOOR_REAR_LEFT = "door_rear_left"
    DOOR_REAR_RIGHT = "door_rear_right"
    HOOD = "hood"
    TRUNK = "trunk"
    FENDER_FRONT_LEFT = "fender_front_left"
    FENDER_FRONT_RIGHT = "fender_front_right"
    FENDER_REAR_LEFT = "fender_rear_left"
    FENDER_REAR_RIGHT = "fender_rear_right"
    HEADLIGHT_LEFT = "headlight_left"
    HEADLIGHT_RIGHT = "headlight_right"


class DamageType(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    RUST = "rust"
    PAINT_CHIP = "paint_chip"


class DamageSource(str, Enum):
    ML = "ml"
    MANUAL = "manual"
