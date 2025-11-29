from enum import Enum


class KafkaTopic(str, Enum):
    SCENARIO_EVENTS = "scenario_events"
