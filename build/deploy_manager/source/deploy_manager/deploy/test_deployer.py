from deploy_manager.common.subprocess_executor import SubprocessExecutor
from deploy_manager.docker.docker_builder import DockerBuilder
from deploy_manager.constants import BASE_IMAGES, PYTHON_BASE, PYTHON_BUILD, SERVICES_SECTION, DeployMode, ServicesType
from deploy_manager.deploy.deployer_abstract import Deployer
from deploy_manager.docker.docker_compose_runner import DockerComposeRunner


class TestDeployer(Deployer):
    def __init__(self,
                 config_data: dict,
                 dc_runner: DockerComposeRunner,
                 executor: SubprocessExecutor,):
        super().__init__(config_data, dc_runner=dc_runner, executor=executor)

    def deploy(self) -> None:
        builder = DockerBuilder(self._executor)

        self._logger.info("Building base python images...")
        self._build_base_images(builder)

        self._logger.info("Building test images...")
        config_services = self.config_data[SERVICES_SECTION]
        test_services = config_services[ServicesType.TEST_SERVICES.value]
        for service_name, service_conf in test_services.items():
            image_name = f"{DeployMode.TEST.value}-{service_name}"
            builder.build_image(service_conf, image_name)

        self._logger.info("Test images built successfully")

    def _build_base_images(self, builder: DockerBuilder) -> None:
        base_images_conf = self.config_data[BASE_IMAGES]
        builder.build_image(base_images_conf[PYTHON_BUILD], f'{PYTHON_BUILD}_image')
        builder.build_image(base_images_conf[PYTHON_BASE], f'{PYTHON_BASE}_image')
