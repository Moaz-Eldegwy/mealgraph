.PHONY: install dev test lint format clean run

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

test:
	pytest -ra -q

test-cov:
	pytest -ra --cov=. --cov-report=term-missing --cov-report=html

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

run:
	python -c "import mealgraph; print('Module imports OK')"
