IMAGE_NAME := hotel-cancellation-api
IMAGE_TAG  := latest

.PHONY: help install train lint format format-check test test-fast build run run-dev clean all

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	PYTHONPATH=. python -m src.train

lint:
	PYTHONPATH=. ruff check src/ app/ tests/

format:
	ruff format src/ app/ tests/
	PYTHONPATH=. ruff check --fix src/ app/ tests/

format-check:
	ruff format --check src/ app/ tests/
	PYTHONPATH=. ruff check src/ app/ tests/

test:
	PYTHONPATH=. pytest tests/ -v --cov=app --cov=src --cov-report=term-missing --cov-fail-under=80

test-fast:
	PYTHONPATH=. pytest tests/ -v -x

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

run:
	docker run --rm -p 8000:8000 $(IMAGE_NAME):$(IMAGE_TAG)

run-dev:
	PYTHONPATH=. uvicorn app.main:app --reload --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov/ dist/

all: install format-check lint test build
