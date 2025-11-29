#!/bin/bash

set -e

export REPO_ROOT=$(pwd)

# rm -rf "$REPO_ROOT/build/whl"
# mkdir -p "$REPO_ROOT/build/whl"

echo "Installing bootstrap..."
pip3 install --default-timeout=3000 --retries=10 "$REPO_ROOT/build/package_bootstrap"

echo "Building packages..."
package-bootstrap

echo "Verifying wheels..."
if [ -z "$(ls -A "$REPO_ROOT/build/whl")" ]; then
    echo "Error: No wheels built!"
    exit 1
fi

echo "Installing builder..."
pip3 install --default-timeout=3000 --retries=10 "$REPO_ROOT"/build/whl/deploy_manager-*.whl

if [[ "$1" == "--test" ]]; then
    run-deploy --test
else
    run-deploy
fi

# [debug] для сборки конкретного пакета
# misis-bootstrap --packages misis_runner

# [debug] сборка контейнеров конкретных сервисов
# TODO: Сделать в misis_builder специальный флаг для сборки одного отдельного сервиса
# docker build -f build/docker/orchestrator_service/Dockerfile -t misis-orchestrator_service . 
# docker build -f build/docker/api_service/Dockerfile -t misis-api_service .
# docker build -f build/docker/runner_service/Dockerfile -t misis-runner_service .