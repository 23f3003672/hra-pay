.PHONY: help install dev test lint fmt env-demo calibrate train eval priors clean

PY ?= python3

help:
	@echo "HRA-Pay targets:"
	@echo "  make install    install the package and runtime deps"
	@echo "  make dev        install with dev + dashboard extras"
	@echo "  make test       run the test suite"
	@echo "  make lint       ruff check"
	@echo "  make fmt        ruff format + autofix"
	@echo "  make env-demo   roll out the environment under a random policy"
	@echo "  make calibrate  run the offline LLM reward calibration"
	@echo "  make train      train the flat DQN"
	@echo "  make eval       evaluate all policies and write results/"
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

calibrate:
	$(PY) -m hrapay.rewards.calibrator

train:
	$(PY) -m hrapay.train --agent flat --steps 60000 --seeds 0 1 2
	$(PY) -m hrapay.train --agent bdq --steps 60000 --seeds 0 1 2

eval:
	$(PY) -m hrapay.eval.cli --episodes 1000 --eval-seeds 3

priors:
	$(PY) -m hrapay.env.demo --episodes 1 --refresh-priors

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info build dist
