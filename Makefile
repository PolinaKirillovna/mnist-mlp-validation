# Makefile for the mnist-mlp-validation project.
# Thin wrappers over the package CLI (python -m mnist_validation ...).

PYTHON ?= python
PKG := mnist_validation
RAW_SRC ?= ../materials/data
RAW_DST := data/raw
CONFIG ?= configs/lab1_base.yaml
GRID ?= configs/lab1_grid.yaml

.PHONY: help install data lint format typecheck test smoke \
        validate train grid interpret jax report notebook all clean

help:
	@echo "Targets:"
	@echo "  install    pip install -e '.[dev]'"
	@echo "  data       copy variant files from $(RAW_SRC) into $(RAW_DST)"
	@echo "  lint       ruff check + ruff format --check"
	@echo "  format     ruff format + ruff check --fix"
	@echo "  typecheck  mypy on src"
	@echo "  test       pytest"
	@echo "  smoke      fast end-to-end run (2 epochs, subsample)"
	@echo "  validate   Part 1: data validation, anomalies, figures, tables"
	@echo "  grid       Part 2: full training grid (torch)"
	@echo "  interpret  Part 3: error analysis, occlusion, first-layer maps"
	@echo "  jax        Part 4: JAX runs and framework comparison"
	@echo "  report     build reports/report.docx via pandoc"
	@echo "  notebook   execute notebooks/lab1_mnist_validation.ipynb"
	@echo "  all        validate -> grid -> interpret -> jax"

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	@mkdir -p $(RAW_DST)
	cp -v $(RAW_SRC)/d3.csv $(RAW_DST)/d3.csv
	cp -v $(RAW_SRC)/mnist_test.csv $(RAW_DST)/mnist_test.csv

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src

test:
	pytest

smoke:
	$(PYTHON) -m $(PKG) smoke --config $(CONFIG)

validate:
	$(PYTHON) -m $(PKG) validate --config $(CONFIG)

grid:
	$(PYTHON) -m $(PKG) grid --config $(GRID)

interpret:
	$(PYTHON) -m $(PKG) interpret --config $(CONFIG)

jax:
	$(PYTHON) -m $(PKG) jax-compare --config $(GRID)

all: validate grid interpret jax

report:
	pandoc reports/report.md \
		--metadata lang=ru \
		--lua-filter=reports/pagebreak.lua \
		--reference-doc=reports/reference.docx \
		--resource-path=reports \
		-o reports/report.docx

notebook:
	jupyter nbconvert --to notebook --execute --inplace \
		notebooks/lab1_mnist_validation.ipynb

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
