from enum import Enum


class ScenarioStatus(str, Enum):
    INIT_STARTUP = "init_startup"
    IN_STARTUP_PROCESSING = "in_startup_processing"
    ACTIVE = "active"
    INIT_SHUTDOWN = "init_shutdown"
    IN_SHUTDOWN_PROCESSING = "in_shutdown_processing"
    INACTIVE = "inactive"
