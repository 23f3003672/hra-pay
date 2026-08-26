.PHONY: help install dev test lint fmt env-demo priors clean

PY ?= python3

help:
	@echo "HRA-Pay targets:"
	@echo "  make install    install the package and runtime deps"
	@echo "  make dev        install with dev + dashboard extras"
	@echo "  make test       run the test suite"
	@echo "  make lint       ruff check"
	@echo "  make fmt        ruff format + autofix"
	@echo "  make env-demo   roll out the environment under a random policy"
	@echo "  make priors     recompute the empirical channel-success priors"

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev,dashboard]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

env-demo:
	$(PY) -m hrapay.env.demo --episodes 5

priors:
	$(PY) -m hrapay.env.demo --episodes 1 --refresh-priors

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info build dist
