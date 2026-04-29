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
    DOOR = "door"
    FRONT_FENDER = "front_fender"
    REAR_FENDER = "rear_fender"
    TRUNK = "trunk"
    HOOD = "hood"
    ROOF = "roof"
    HEADLIGHT = "headlight"
    FRONT_WINDSHIELD = "front_windshield"
    REAR_WINDSHIELD = "rear_windshield"
    SIDE_WINDOW = "side_window"
    WHEEL = "wheel"
    BUMPER = "bumper"


class DamageType(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    PAINT_CHIP = "paint_chip"
    RUST = "rust"
    CRACK = "crack"
    BROKEN_GLASS = "broken_glass"
    FLAT_TIRE = "flat_tire"
    BROKEN_HEADLIGHT = "broken_headlight"


class DamageSource(str, Enum):
    ML = "ml"
    MANUAL = "manual"
