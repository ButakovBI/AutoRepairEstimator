import logging

from deploy_manager.common.subprocess_executor import SubprocessExecutor
from deploy_manager.constants import DOCKERFILE, LIBS, PARENT_IMAGE, PROJECT_ROOT


class DockerBuilder:
    def __init__(self, executor: SubprocessExecutor):
        self._executor = executor

    def build_image(self, config_data: dict, image_name: str) -> None:
        dockerfile_path = PROJECT_ROOT / config_data[DOCKERFILE]
        cmd = ['docker', 'build', '-f', str(dockerfile_path), '-t', image_name]

        build_args = self._prepare_build_args(config_data)
        for arg in build_args:
            cmd += ['--build-arg', arg]

        cmd.append(str(PROJECT_ROOT))
        self._logger.info(f'Building image {image_name}...')
        error_context = f'Error while building image {image_name}'
        self._executor.execute_cmd(cmd, error_context)

    def _prepare_build_args(self, config_data: dict) -> list[str]:
        args: list[str] = []
        added_libs = ' '.join(config_data.get(LIBS, []))
        parent_image = config_data.get(PARENT_IMAGE, None)
        if parent_image:
            args.append(f'{PARENT_IMAGE}={parent_image}')
        if added_libs:
            args.append(f'{LIBS}={added_libs}')
        return args

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
