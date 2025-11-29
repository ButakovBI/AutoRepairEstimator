FROM python:3.12-slim

ARG VENV_PATH=/opt/venv/python_build

RUN apt-get update \
    && apt-get install -y ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv ${VENV_PATH}

COPY build/docker/python_build/pip.conf ${VENV_PATH}/pip.conf
COPY build/docker/python_build/entrypoint.sh /entrypoint.sh

ENV VENV_PATH=${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

ENTRYPOINT ["/entrypoint.sh"]
