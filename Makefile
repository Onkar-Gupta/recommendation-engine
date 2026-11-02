# Unix/macOS/WSL. On Windows without make, use run.ps1 (same targets).
PY = venv/bin/python

.PHONY: setup data train train-custom serve test lint all docker-build docker-run

setup:
	python3 -m venv venv
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

train:
	$(PY) -m src.train --config config.yaml

# make train-custom DATA=path/to/my.csv
train-custom:
	$(PY) -m src.train --config config.yaml --data $(DATA)

serve:
	$(PY) -m uvicorn src.serve:create_app --factory --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest tests/

lint:
	$(PY) -m ruff check .

# Fresh clone to running API. `serve` blocks, so curl from a second terminal.
all: setup data train serve

docker-build:
	docker build -t recommendation-engine .

# The image holds no model or data: it reads the ones you trained locally.
docker-run:
	docker run --rm -p 8000:8000 -v "$(CURDIR)/models:/app/models:ro" -v "$(CURDIR)/data:/app/data:ro" recommendation-engine
