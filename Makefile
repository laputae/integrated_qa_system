.PHONY: install lint format test docker-build docker-up docker-down

# ===== 开发环境 =====

install:
	uv sync --frozen --group dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest tests/ -v --ignore=tests/test_gpu_fp16.py -k "not milvus"

# ===== Docker =====

docker-build:
	docker build -t integrated-qa-system:latest .

docker-up:
	docker compose --profile dev up -d

docker-down:
	docker compose --profile dev down

docker-logs:
	docker compose logs -f app
