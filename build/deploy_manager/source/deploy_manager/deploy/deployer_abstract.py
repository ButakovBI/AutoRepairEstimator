import logging
from abc import ABC, abstractmethod

from deploy_manager.common.subprocess_executor import SubprocessExecutor
from deploy_manager.docker.docker_compose_runner import DockerComposeRunner


class Deployer(ABC):
    def __init__(self,
                 config_data: dict,
                 dc_runner: DockerComposeRunner,
                 executor: SubprocessExecutor,):
        self.config_data = config_data
        self._executor = executor
        self._dc_runner = dc_runner

    @abstractmethod
    def deploy(self) -> None:
        pass

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
