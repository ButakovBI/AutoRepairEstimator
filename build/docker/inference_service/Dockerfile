ARG PARENT_IMAGE

FROM $PARENT_IMAGE AS builder

ARG LIBS

RUN pip3 install $LIBS --find-links /whl && rm -rf /whl

FROM python:3.12-slim

ENV VENV_PATH=/opt/venv/python_build
ENV PATH="${VENV_PATH}/bin:$PATH"
COPY --from=builder ${VENV_PATH} ${VENV_PATH}

HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8001/docs || exit 1

CMD ["uvicorn", "misis_inference.app.web.app:app", "--host", "0.0.0.0", "--port", "8001"]
