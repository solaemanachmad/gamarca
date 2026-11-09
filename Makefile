.PHONY: help install test lint clean

PYTHON ?= python3

help:        ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:     ## Install package dependencies
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:        ## Run unit and golden-snapshot tests (no GPU/API needed)
	@$(PYTHON) tests/test_fingerprint.py | tail -1
	@$(PYTHON) tests/test_calibrated.py | tail -1
	@$(PYTHON) tests/test_logpre.py | tail -1
	@$(PYTHON) tests/test_gama.py | tail -1
	@$(PYTHON) tests/test_aiops2025.py | tail -1
	@$(PYTHON) tests/test_api_client.py | tail -1
	@$(PYTHON) tests/test_v6.py | tail -1

lint:        ## Run pyflakes linter on active code
	$(PYTHON) -m pyflakes $$(git ls-files '*.py' 2>/dev/null || find . -name "*.py" -not -path "*/archive/*" -not -path "*/.venv/*")

clean:       ## Remove Python cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
