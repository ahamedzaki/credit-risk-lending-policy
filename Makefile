.PHONY: help venv install synth preflight pipeline app test clean

PY := .venv/bin/python
PIP := .venv/bin/pip

help:
	@echo "make install    - create .venv (needs python3.11) and install requirements"
	@echo "make synth      - generate data/synthetic_accepted.csv for local testing"
	@echo "make preflight  - run the maturity pre-flight (needs data/ CSV + config)"
	@echo "make pipeline   - run all stages: raw..check"
	@echo "make app        - launch the Streamlit simulator"
	@echo "make test       - run unit tests"
	@echo "make clean      - remove the built DB, exports and figures"

venv:
	python3.11 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

synth:
	$(PY) scripts/make_synthetic_data.py --rows 12000 --out data/synthetic_accepted.csv

preflight:
	$(PY) run_pipeline.py preflight

pipeline:
	$(PY) run_pipeline.py all

app:
	.venv/bin/streamlit run app/simulator.py

test:
	$(PY) tests/test_sim_core.py

clean:
	rm -f data/credit.duckdb data/credit.duckdb.wal exports/*.parquet reports/figures/*.png
	@echo "cleaned build outputs (raw CSV and committed sample kept)"
