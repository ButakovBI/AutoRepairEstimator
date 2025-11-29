import logging
import subprocess

from deploy_manager.constants import ERROR


class SubprocessExecutor:
    def execute_cmd(self, cmd: list[str], error_context: str = ERROR):
        cmd_str = ' '.join(str(c) for c in cmd)

        try:
            self._logger.info(f"Executing command: {cmd_str}")
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            msg = f"{error_context}: Command failed with code {e.returncode}. Command: {cmd_str}"
            self._logger.error(msg)
            raise RuntimeError(msg)
        except Exception as ex:
            msg = f"{error_context}: Unexpected error: {ex}. Command: {cmd_str}"
            self._logger.error(msg)
            raise RuntimeError(msg)

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
