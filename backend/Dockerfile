# syntax=docker/dockerfile:1
# trixie (Debian 13): SuiteSparse 7.x — the only distro line whose
# cholmod_common carries both field generations (dbound/ndbounds_hit AND
# sbound/nsbounds_hit) that scikit-sparse's compat .pxd can compile
# against. bookworm's SS 5.12 cannot build it.
FROM python:3.12-slim-trixie AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYVISTA_OFF_SCREEN=true \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libegl1 libopengl0 libxrender1 libgomp1 libglu1-mesa curl \
      libsuitesparse-dev \
      libharfbuzz0b \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY alembic.ini ./
COPY alembic ./alembic
COPY cadx ./cadx
# --extra repair pulls optional pymeshlab (GPL) — the mesh repair/remesh
# engine in cad/mesh_repair_pml.py; keeps preview (worker) and executor
# ops on the same code path
RUN uv sync --frozen --no-dev --extra repair

ENV PATH="/app/.venv/bin:$PATH"

FROM base AS api
CMD ["sh", "-c", "alembic upgrade head && uvicorn cadx.main:app --host 0.0.0.0 --port 8000"]

FROM base AS worker
CMD ["arq", "cadx.queue.AgentWorkerSettings"]

FROM base AS executor
# FEM solver tier lands only in the executor image: scikit-sparse is a
# git-pinned sdist (see pyproject) built against libsuitesparse-dev (in
# base), pyamg builds a pybind11 extension on platforms without wheels —
# both need a compiler, and uv needs git to fetch the pinned repo.
# All build tools are purged inside this one layer so the image stays
# slim. solver.py falls back to SuperLU/SSOR-CG when either import fails,
# so the extra is strictly additive.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && uv sync --frozen --no-dev --extra repair --extra fem \
    && apt-get purge -y build-essential git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd -r cad && useradd -r -g cad -u 1000 -m cad \
    && mkdir -p /data && chown -R cad:cad /data
USER cad
ENV PYTHONFAULTHANDLER=1
# diag wrapper: logs incoming signals and arq's exit code so silent
# clean-exit restarts reveal whether docker or the process ended it
CMD ["sh", "-c", "trap 'echo \"[diag] got SIGTERM, forwarding\" >&2; kill -TERM $C' TERM; trap 'echo \"[diag] got SIGINT, forwarding\" >&2; kill -INT $C' INT; arq cadx.queue.CadWorkerSettings & C=$!; rc=0; while kill -0 $C 2>/dev/null; do wait $C; rc=$?; done; echo \"[diag] arq exited rc=$rc\" >&2; exit $rc"]

# test variant of the executor image: same env + dev deps, for
# `docker build --target executor-test` then run pytest tests/cad inside it.
FROM executor AS executor-test
USER root
RUN uv sync --frozen --extra repair --extra fem
USER cad
