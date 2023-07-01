setup:
	pip install poetry
	poetry install

format:
	poetry run black peptriever

check-code: lint-python check-formatting static-type-check

check-formatting:
	poetry run black --check peptriever

static-type-check:
	poetry run mypy --install-types --non-interactive peptriever
	poetry run mypy -p peptriever
	find . -type f -name "*.sh" | xargs poetry run shellcheck

lint-python:
	poetry run ruff peptriever

test:
	poetry run pytest peptriever