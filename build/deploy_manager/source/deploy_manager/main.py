import argparse
import json

from deploy_manager.common.subprocess_executor import SubprocessExecutor
from deploy_manager.constants import DEFAULT_COMPOSE_PATH, DEFAULT_IMAGES_CONF_PATH
from deploy_manager.deploy.prod_deployer import ProdDeployer
from deploy_manager.deploy.test_deployer import TestDeployer
from deploy_manager.docker.docker_compose_runner import DockerComposeRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='Build test images and run all tests')
    parser.add_argument('--compose-file', default=DEFAULT_COMPOSE_PATH,
                        help='Path to docker compose file')
    parser.add_argument('--images-config', default=DEFAULT_IMAGES_CONF_PATH,
                        help='Path to docker images configuration')
    args = parser.parse_args()
    with open(args.images_config, 'r') as f:
        config_data = json.load(f)

    executor = SubprocessExecutor()
    dc_runner = DockerComposeRunner(str(args.compose_file), executor=executor)

    if args.test:
        TestDeployer(config_data=config_data, dc_runner=dc_runner, executor=executor).deploy()
    else:
        ProdDeployer(config_data=config_data, dc_runner=dc_runner, executor=executor).deploy()
