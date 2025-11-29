from enum import Enum


class KafkaTopic(str, Enum):
    HEARTBEATS = "heartbeats"
    SCENARIO_EVENTS = "scenario_events"
    RUNNER_COMMANDS = "runner_commands"
