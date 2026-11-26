format:
	@uv run black .
	@uv run isort .

check:
	@uv run black --check .
	@uv run isort --check-only .
	@uv run pytest

test:
	@uv run pytest

install:
	@uv sync --group dev
