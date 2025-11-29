from deploy_manager.common.subprocess_executor import SubprocessExecutor
from deploy_manager.docker.docker_builder import DockerBuilder
from deploy_manager.constants import BASE_IMAGES, PYTHON_BASE, PYTHON_BUILD, SERVICES_SECTION, DeployMode, ServicesType
from deploy_manager.deploy.deployer_abstract import Deployer, DockerComposeRunner


class ProdDeployer(Deployer):
    def __init__(self,
                 config_data: dict,
                 dc_runner: DockerComposeRunner,
                 executor: SubprocessExecutor,):
        super().__init__(config_data, dc_runner=dc_runner, executor=executor)

    def deploy(self) -> None:
        builder = DockerBuilder(self._executor)

        self._logger.info("Building base python images...")
        self._build_base_images(builder)

        self._logger.info("Deploying production services...")
        config_services = self.config_data[SERVICES_SECTION]
        prod_services = config_services[ServicesType.PROD_SERVICES.value]
        for service_name, service_conf in prod_services.items():
            image_name = f"{DeployMode.PROD.value}-{service_name}"
            builder.build_image(service_conf, image_name)

        self._dc_runner.up_compose(daemon=True)

        self._logger.info("Production services deployed successfully")

    def _build_base_images(self, builder: DockerBuilder) -> None:
        base_images_conf = self.config_data[BASE_IMAGES]
        builder.build_image(base_images_conf[PYTHON_BUILD], f'{PYTHON_BUILD}_image')
        builder.build_image(base_images_conf[PYTHON_BASE], f'{PYTHON_BASE}_image')
