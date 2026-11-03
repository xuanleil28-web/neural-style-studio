PYTHON ?= python

.PHONY: install run test lint format docker-build docker-run clean

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

run:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

docker-build:
	docker build -t neural-style-studio .

docker-run:
	docker run --rm -p 8501:8501 neural-style-studio

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache outputs
