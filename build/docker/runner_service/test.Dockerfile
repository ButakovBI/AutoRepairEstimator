ARG PARENT_IMAGE

FROM $PARENT_IMAGE AS builder

ARG LIBS

RUN pip3 install $LIBS --find-links /whl && rm -rf /whl

FROM python:3.12-slim

ENV VENV_PATH=/opt/venv/python_build
ENV PATH="${VENV_PATH}/bin:$PATH"
COPY --from=builder ${VENV_PATH} ${VENV_PATH}

CMD ["pytest", "-q"]
