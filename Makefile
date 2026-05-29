.PHONY: help install ingest clean queries ml ml-price ml-occupancy bench bench-local bench-summary dash test lint format clean-data

PYTHON ?= python
CITIES ?= porto,lisbon,madrid,barcelona
QUERY  ?= all
CORES  ?= 4
JAVA_HOME ?= $(shell /usr/libexec/java_home -v 17 2>/dev/null || printf "%s/libexec/openjdk.jdk/Contents/Home" "$$(brew --prefix openjdk@17 2>/dev/null)")
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)

help:
	@echo "Targets:"
	@echo "  install      Install package + dev/dashboard extras"
	@echo "  ingest       Read data/raw/ CSVs → data/interim/ unified Parquet"
	@echo "  clean        data/interim/ → data/processed/ (partitioned by city)"
	@echo "  queries      Run all queries → reports/results/qNN.parquet"
	@echo "  queries QUERY=q03   Run a single query"
	@echo "  ml           Train both ML pipelines (price + occupancy)"
	@echo "  ml-price     Train price regression only"
	@echo "  ml-occupancy Train occupancy classifier only"
	@echo "  bench-local  Run benchmark workloads at local[1,2,4,8]"
	@echo "  bench-summary Parse Dataproc benchmark logs -> reports/benchmarks/summary.csv"
	@echo "  dash         streamlit run dashboard/app.py"
	@echo "  test         pytest"
	@echo "  lint         ruff check"

install:
	$(PYTHON) -m pip install -e ".[dev,dashboard]"

ingest:
	$(PYTHON) -m jobs.ingest --cities $(CITIES)

clean:
	$(PYTHON) -m jobs.clean --cities $(CITIES)

queries:
	$(PYTHON) -m jobs.run_queries --query $(QUERY)

ml: ml-price ml-occupancy ml-monthly

ml-price:
	$(PYTHON) -m jobs.train_price_model

ml-occupancy:
	$(PYTHON) -m jobs.train_occupancy_model

ml-monthly:
	$(PYTHON) -m jobs.train_monthly_occupancy

importance:
	$(PYTHON) -m jobs.feature_importance

bench-local:
	$(PYTHON) -m jobs.benchmark --platform local --workload query_seasonality --runs 3
	$(PYTHON) -m jobs.benchmark --platform local --workload train_rf --runs 3

bench-summary:
	$(PYTHON) -m jobs.summarize_benchmarks

dash:
	$(PYTHON) -m streamlit run dashboard/app.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests jobs dashboard

format:
	$(PYTHON) -m ruff format src tests jobs dashboard

clean-data:
	@echo "Removing data/interim and data/processed (data/raw/ untouched)"
	rm -rf data/interim/* data/processed/*
