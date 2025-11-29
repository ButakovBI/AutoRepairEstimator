import logging

from deploy_manager.common.subprocess_executor import SubprocessExecutor


class DockerComposeRunner:
    DOWN_COMMAND = 'down'
    UP_COMMAND = 'up'

    def __init__(self, compose_file: str, executor: SubprocessExecutor):
        self._base_cmd = ['docker-compose', '-f', compose_file]
        self._executor = executor

    def down_compose(self) -> None:
        cmd = self._base_cmd + [self.DOWN_COMMAND]
        self._executor.execute_cmd(cmd, f'Docker compose {self.DOWN_COMMAND} error')

    def up_compose(self, daemon: bool = True) -> None:
        cmd = self._base_cmd + [self.UP_COMMAND]
        if daemon:
            cmd.append('-d')
        self._executor.execute_cmd(cmd, f'Docker compose {self.UP_COMMAND} error')

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
