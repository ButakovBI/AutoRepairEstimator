from enum import Enum


class KafkaTopic(str, Enum):
    HEARTBEATS = "heartbeats"
    RUNNER_COMMANDS = "runner_commands"
