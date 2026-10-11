# cadagent — LLM-driven CAD agent for 3D printing & CNC
# Requires: uv (python -m pip install uv), node >= 20, docker compose

VENV := .venv
PY := $(VENV)/bin/python
BACKEND := backend
FRONTEND := frontend

.PHONY: setup lock sync dev dev-api dev-worker dev-executor dev-web infra \
        up down logs test lint fmt eval migrate build clean \
        mesh-sync mesh-server mesh-models mesh-test dev-mesh

setup: $(VENV)/bin/uv
$(VENV)/bin/uv:
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install uv

lock: setup
	cd $(BACKEND) && ../$(VENV)/bin/uv lock

sync: lock
	cd $(BACKEND) && ../$(VENV)/bin/uv sync

## Local dev (infra in docker, apps native)
infra:
	docker compose -f docker-compose.dev.yml up -d postgres redis

dev-api:
	cd $(BACKEND) && ../$(VENV)/bin/uv run alembic upgrade head && ../$(VENV)/bin/uv run uvicorn cadagent.main:app --reload --port 8000

dev-worker:
	cd $(BACKEND) && ../$(VENV)/bin/uv run arq cadagent.queue.AgentWorkerSettings

dev-executor:
	cd $(BACKEND) && ../$(VENV)/bin/uv run arq cadagent.queue.CadWorkerSettings

dev-web:
	cd $(FRONTEND) && npm run dev

## Full stack in docker
up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

## Quality
test:
	cd $(BACKEND) && ../$(VENV)/bin/uv run pytest tests -x -q

lint:
	cd $(BACKEND) && ../$(VENV)/bin/uv run ruff check .
	cd $(BACKEND) && ../$(VENV)/bin/uv run ruff format --check .
	cd $(FRONTEND) && npx tsc --noEmit

fmt:
	cd $(BACKEND) && ../$(VENV)/bin/uv run ruff check --fix . && ../$(VENV)/bin/uv run ruff format .

migrate:
	cd $(BACKEND) && ../$(VENV)/bin/uv run alembic upgrade head

eval:
	cd $(BACKEND) && ../$(VENV)/bin/uv run cadagent eval

build:
	docker compose build

## One-command setup & start (GPU auto-detection)
start:
	bash scripts/start.sh

start-dev:
	bash scripts/start.sh --dev

stop:
	bash scripts/stop.sh

status:
	bash scripts/status.sh

autosetup:
	bash scripts/setup.sh --dev

doctor:
	bash scripts/setup.sh --doctor

## Mesh generator sidecar (native, own venv — see mesh_server/README.md)
MESH := mesh_server
MESHVENV := $(MESH)/.venv
# platform extras: mlx on arm64 macs, nothing heavy elsewhere (install adapters via scripts)
MESH_EXTRA := $(shell [ "$$(uname -sm)" = "Darwin arm64" ] && echo ".[bg]" || echo ".")

mesh-sync: setup
	cd $(MESH) && ../$(VENV)/bin/uv venv $(VENV)
	cd $(MESH) && ../$(VENV)/bin/uv pip install -e '$(MESH_EXTRA)' --python $(VENV)/bin/python
	cd $(MESH) && ../$(VENV)/bin/uv pip install --group dev --python $(VENV)/bin/python

mesh-server:
	cd $(MESH) && $(VENV)/bin/meshserver serve

dev-mesh: mesh-server

mesh-models:
	cd $(MESH) && $(VENV)/bin/meshserver download auto

mesh-test:
	cd $(MESH) && $(VENV)/bin/python -m pytest tests -x -q

clean:
	rm -rf $(BACKEND)/.venv $(FRONTEND)/node_modules $(FRONTEND)/dist .mypy_cache .pytest_cache .ruff_cache
