# syntax=docker/dockerfile:1
# trixie (Debian 13): SuiteSparse 7.x — the only distro line whose
# cholmod_common carries both field generations (dbound/ndbounds_hit AND
# sbound/nsbounds_hit) that scikit-sparse's compat .pxd can compile
# against. bookworm's SS 5.12 cannot build it.

# cadx_native wheel: built on the SAME trixie base so glibc matches the
# runtime images. Rust only exists in this stage — the services ship the
# prebuilt wheel. The module is optional (cad/cad/native.py falls back),
# but always built here so parity tests can gate its behavior.
FROM python:3.12-slim-trixie AS rust-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && pip install --no-cache-dir maturin
WORKDIR /native
COPY cadx_native/Cargo.toml cadx_native/Cargo.lock* ./
COPY cadx_native/pyproject.toml ./
COPY cadx_native/src ./src
RUN PATH=/root/.cargo/bin:$PATH maturin build --release -i python3.12 -o dist

FROM python:3.12-slim-trixie AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYVISTA_OFF_SCREEN=true \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libegl1 libopengl0 libxrender1 libgomp1 libglu1-mesa curl \
      libxcursor1 \
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
# wheel install AFTER the stage's last `uv sync` — sync removes packages
# that are not in uv.lock, and cadx_native is a local wheel, not a lock dep
COPY --from=rust-builder /native/dist/*.whl /tmp/nw/
RUN uv pip install --python /app/.venv/bin/python /tmp/nw/*.whl && rm -rf /tmp/nw

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
# re-install after this stage's own `uv sync` wiped the base-layer copy
COPY --from=rust-builder /native/dist/*.whl /tmp/nw/
RUN uv pip install --python /app/.venv/bin/python /tmp/nw/*.whl && rm -rf /tmp/nw
RUN groupadd -r cad && useradd -r -g cad -u 1000 -m cad \
    && mkdir -p /data && chown -R cad:cad /data
USER cad
ENV PYTHONFAULTHANDLER=1
# per-lane supervisor: each arq worker is respawned INDEPENDENTLY — a dying
# lane (transient OCC/segfault exit) must never kill a multi-hour SIMP on a
# sibling lane (the old restart-all policy cancelled robot_arm r23's topopt
# at iteration 12). Signals forward to all three; STOP stops respawning.
CMD ["sh", "-c", "S=''; trap 'echo \"[diag] got SIGTERM, forwarding\" >&2; S=1; kill -TERM $C1 $C2 $C3 2>/dev/null' TERM; trap 'echo \"[diag] got SIGINT, forwarding\" >&2; S=1; kill -INT $C1 $C2 $C3 2>/dev/null' INT; arq cadx.queue.CadWorkerSettings & C1=$!; arq cadx.queue.CadLightWorkerSettings & C2=$!; arq cadx.queue.CadFemWorkerSettings & C3=$!; while [ -z \"$S\" ]; do kill -0 $C1 2>/dev/null || { echo \"[diag] cad lane pid=$C1 died — respawning cad only\" >&2; arq cadx.queue.CadWorkerSettings & C1=$!; }; kill -0 $C2 2>/dev/null || { echo \"[diag] cad-light lane pid=$C2 died — respawning cad-light only\" >&2; arq cadx.queue.CadLightWorkerSettings & C2=$!; }; kill -0 $C3 2>/dev/null || { echo \"[diag] cad-fem lane pid=$C3 died — respawning cad-fem only\" >&2; arq cadx.queue.CadFemWorkerSettings & C3=$!; }; sleep 2; done; wait $C1 $C2 $C3 2>/dev/null; exit 0"]

# test variant of the executor image: same env + dev deps, for
# `docker build --target executor-test` then run pytest tests/cad inside it.
FROM executor AS executor-test
USER root
RUN uv sync --frozen --extra repair --extra fem
COPY --from=rust-builder /native/dist/*.whl /tmp/nw/
RUN uv pip install --python /app/.venv/bin/python /tmp/nw/*.whl && rm -rf /tmp/nw
USER cad
