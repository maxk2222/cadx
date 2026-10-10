# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYVISTA_OFF_SCREEN=true \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libegl1 libopengl0 libxrender1 libgomp1 libglu1-mesa curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY alembic.ini ./
COPY alembic ./alembic
COPY cadagent ./cadagent
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

FROM base AS api
CMD ["sh", "-c", "alembic upgrade head && uvicorn cadagent.main:app --host 0.0.0.0 --port 8000"]

FROM base AS worker
CMD ["arq", "cadagent.queue.AgentWorkerSettings"]

FROM base AS executor
RUN groupadd -r cad && useradd -r -g cad -u 1000 -m cad \
    && mkdir -p /data && chown -R cad:cad /data
USER cad
CMD ["arq", "cadagent.queue.CadWorkerSettings"]
