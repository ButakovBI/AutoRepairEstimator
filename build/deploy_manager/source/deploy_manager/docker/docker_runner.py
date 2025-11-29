import logging

from deploy_manager.common.subprocess_executor import SubprocessExecutor


class DockerRunner:
    def __init__(self, executor: SubprocessExecutor):
        self._executor = executor

    def run_container(self, image: str):
        cmd = ['docker', 'run', '--rm', image]
        self._executor.execute_cmd(cmd, 'Run container error')

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
