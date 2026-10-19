# CADX — LLM-driven CAD agent for 3D printing & CNC
# Requires: uv (python -m pip install uv), node >= 20, docker compose
#
# `make` or `make help` lists all targets.

.DEFAULT_GOAL := help

VENV := .venv
PY := $(VENV)/bin/python
BACKEND := backend
FRONTEND := frontend

.PHONY: help setup lock sync dev-api dev-worker dev-executor dev-web infra \
        up down logs logs-dev logs-api logs-worker logs-executor logs-web \
        test lint fmt eval eval-real migrate build clean clean-all \
        mesh-sync mesh-server mesh-models mesh-test dev-mesh \
        start start-dev stop status restart restart-dev autosetup doctor \
        open update

##@ Setup & run (one command)

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mCADX — make targets\033[0m\n"} \
	  /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2} \
	  /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
	  END {print ""}' $(MAKEFILE_LIST)

start: ## Detect GPU, set up if needed, start full stack (docker + native mesh-server)
	bash scripts/start.sh

start-dev: ## Same but native dev mode (api/worker/executor/vite as processes)
	bash scripts/start.sh --dev

stop: ## Stop everything started by start.sh
	bash scripts/stop.sh

restart: ## stop + start (docker mode)
	bash scripts/stop.sh && bash scripts/start.sh

restart-dev: ## stop + start (native dev mode)
	bash scripts/stop.sh && bash scripts/start.sh --dev

status: ## Health of all components (--watch/--json supported via script)
	bash scripts/status.sh

open: ## Open the app in the browser
	@([ "$$(uname -s)" = "Darwin" ] && open http://localhost:8080) \
	  || xdg-open http://localhost:8080 2>/dev/null || true

autosetup: ## Re-run the installer (GPU detection + adapters + dev deps)
	bash scripts/setup.sh --dev

doctor: ## Diagnostics: tier, adapters, tool versions, disk, ports
	bash scripts/setup.sh --doctor

update: ## git pull --ff-only, then re-run the installer
	git pull --ff-only && bash scripts/setup.sh --dev

##@ Development

setup: $(VENV)/bin/uv
$(VENV)/bin/uv:
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install uv

lock: setup ## Resolve backend dependencies
	cd $(BACKEND) && ../$(VENV)/bin/uv lock

sync: lock ## Install backend dependencies into backend/.venv
	cd $(BACKEND) && ../$(VENV)/bin/uv sync

infra: ## Postgres + Redis in docker for native dev
	docker compose -f docker-compose.dev.yml up -d postgres redis

dev-api: ## FastAPI on :8000 with reload
	cd $(BACKEND) && ../$(VENV)/bin/uv run alembic upgrade head && ../$(VENV)/bin/uv run uvicorn cadx.main:app --reload --port 8000

dev-worker: ## Agent queue worker
	cd $(BACKEND) && ../$(VENV)/bin/uv run arq cadx.queue.AgentWorkerSettings

dev-executor: ## CAD queue worker
	cd $(BACKEND) && ../$(VENV)/bin/uv run arq cadx.queue.CadWorkerSettings

dev-web: ## Vite dev server on :5173
	cd $(FRONTEND) && npm run dev

##@ Docker stack

up: ## Full stack in docker (build + detach)
	docker compose up --build -d

down: ## Stop the docker stack
	docker compose down

logs: ## Tail docker stack logs
	docker compose logs -f --tail=200

logs-api: ## Tail api container logs
	docker compose logs -f --tail=200 api

logs-worker: ## Tail worker container logs
	docker compose logs -f --tail=200 worker

logs-executor: ## Tail cad-executor container logs
	docker compose logs -f --tail=200 cad-executor

logs-web: ## Tail frontend container logs
	docker compose logs -f --tail=200 frontend

logs-dev: ## Tail native dev-mode logs (logs/*.log from start.sh --dev)
	tail -f logs/*.log

build: ## Build all docker images
	docker compose build

##@ Quality

test: ## Backend test suite
	cd $(BACKEND) && ../$(VENV)/bin/uv run pytest tests -x -q

lint: ## ruff + frontend typecheck
	cd $(BACKEND) && ../$(VENV)/bin/uv run ruff check .
	cd $(BACKEND) && ../$(VENV)/bin/uv run ruff format --check .
	cd $(FRONTEND) && npx tsc --noEmit

fmt: ## ruff autofix + format
	cd $(BACKEND) && ../$(VENV)/bin/uv run ruff check --fix . && ../$(VENV)/bin/uv run ruff format .

migrate: ## alembic upgrade head
	cd $(BACKEND) && ../$(VENV)/bin/uv run alembic upgrade head

eval: ## Scripted eval benchmarks
	cd $(BACKEND) && ../$(VENV)/bin/uv run cadx eval

eval-real: ## Eval against the real LLM/mesh stack
	cd $(BACKEND) && ../$(VENV)/bin/uv run cadx eval --real

##@ Mesh sidecar

MESH := mesh_server
MESHVENV := $(MESH)/.venv
# platform extras: bg+llm on arm64 macs (mlx-lm ships the local-LLM adapters),
# nothing heavy elsewhere (install adapters via scripts)
MESH_EXTRA := $(shell [ "$$(uname -sm)" = "Darwin arm64" ] && echo ".[bg,llm]" || echo ".")

mesh-sync: setup ## uv venv + deps for mesh_server/ (+ .venv-llm on Apple Silicon)
	cd $(MESH) && ../$(VENV)/bin/uv venv $(VENV)
	cd $(MESH) && ../$(VENV)/bin/uv pip install -e '$(MESH_EXTRA)' --python $(VENV)/bin/python
	cd $(MESH) && ../$(VENV)/bin/uv pip install --group dev --python $(VENV)/bin/python
	[ "$$(uname -sm)" = "Darwin arm64" ] && ./$(MESH)/scripts/install_llm.sh || true

mesh-server: ## Run the mesh-server on :8765 (foreground)
	cd $(MESH) && $(VENV)/bin/meshserver serve

dev-mesh: mesh-server

mesh-models: ## Download model weights for detected adapters
	cd $(MESH) && $(VENV)/bin/meshserver download auto

mesh-test: ## mesh_server pytest suite
	cd $(MESH) && $(VENV)/bin/python -m pytest tests -x -q

##@ Cleanup

clean: ## Remove backend venv, node_modules, caches
	rm -rf $(BACKEND)/.venv $(FRONTEND)/node_modules $(FRONTEND)/dist .mypy_cache .pytest_cache .ruff_cache

clean-all: ## clean + mesh venv/sources, pidfiles, logs (keeps data/)
	rm -rf $(BACKEND)/.venv $(FRONTEND)/node_modules $(FRONTEND)/dist .mypy_cache .pytest_cache .ruff_cache \
	  $(MESH)/.venv $(MESH)/.src run/*.pid logs/*.log
