.DEFAULT_GOAL := help
.PHONY: help setup check-config infra run up build down logs lint pre-commit test unit-test integration-test e2e-test
TEST_PATH ?= tests
COMPOSE_PROJECT ?= etl-agent
COMPOSE = docker compose --project-directory "$(CURDIR)" -p "$(COMPOSE_PROJECT)" -f "$(CURDIR)/compose.yaml"

help:
	@echo 'setup | infra | run | up | build | down | logs | test | unit-test | integration-test | e2e-test | lint | pre-commit'

setup:
	uv sync --locked
	uv run --locked pre-commit install

check-config:
	uv run --locked python -m etl_agent.startup

run:
	uv run --locked python -m etl_agent

up: check-config
	$(COMPOSE) up --remove-orphans --build -d --wait --wait-timeout 180

infra:
	$(COMPOSE) up -d --wait --wait-timeout 60 postgres

build:
	$(COMPOSE) build api

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked mypy

pre-commit:
	uv run --locked pre-commit run --all-files

test:
	uv run --locked pytest $(TEST_PATH) -q

unit-test:
	$(MAKE) test TEST_PATH=tests/unit

integration-test:
	$(MAKE) test TEST_PATH=tests/integration

e2e-test:
	$(MAKE) test TEST_PATH=tests/e2e
