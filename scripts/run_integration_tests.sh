#!/usr/bin/env bash
# Helper: start a disposable Postgres for the integration test suite.
#
# This script is a convenience wrapper around docker-compose for machines
# where ``testcontainers`` cannot auto-start a container (rootless mode,
# permissions, etc). It:
#
#   1. Brings up ONLY the postgres service from docker/docker-compose.yml
#      so no Kafka / MinIO overhead.
#   2. Waits for the healthcheck to go green.
#   3. Exports INTEGRATION_DB_URL so the conftest picks it up.
#   4. Invokes pytest on the integration suite.
#   5. Tears the container down on exit.
#
# Usage:
#   ./scripts/run_integration_tests.sh                 # full suite
#   ./scripts/run_integration_tests.sh tests/integration/backend/api
#
# Requires Docker Desktop / Podman with the docker CLI shim installed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker/docker-compose.yml"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker CLI not found. Install Docker Desktop or set INTEGRATION_DB_URL directly." >&2
    exit 1
fi

echo ">> starting postgres container..."
docker compose -f "${COMPOSE_FILE}" up -d postgres

cleanup() {
    echo ">> stopping postgres container..."
    docker compose -f "${COMPOSE_FILE}" down
}
trap cleanup EXIT

echo ">> waiting for postgres healthcheck..."
for i in {1..30}; do
    status="$(docker inspect -f '{{.State.Health.Status}}' auto-repair-postgres 2>/dev/null || echo 'starting')"
    if [ "${status}" = "healthy" ]; then
        break
    fi
    sleep 1
done

if [ "${status}" != "healthy" ]; then
    echo "ERROR: postgres did not become healthy within 30s" >&2
    exit 1
fi

export INTEGRATION_DB_URL="postgresql://auto_repair:auto_repair@localhost:5432/auto_repair"
echo ">> INTEGRATION_DB_URL=${INTEGRATION_DB_URL}"

cd "${ROOT_DIR}"
python -m pytest "${@:-tests/integration}" "${@:1}" -v
