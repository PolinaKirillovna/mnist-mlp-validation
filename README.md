# mnist-mlp-validation

[![CI](https://github.com/PolinaKirillovna/mnist-mlp-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/PolinaKirillovna/mnist-mlp-validation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Data validation, two-layer perceptron training in **PyTorch and JAX**, and model
interpretation (occlusion, first-layer weight maps, gradient-based saliency) on an
MNIST subset. The project follows a clean, layered `src`-layout package.

> Educational project for the ITMO course *AI Systems Validation* (laboratory
> practicum, 2024). Report and notebook are in Russian; code and docs are in English.

## What is done

Work is organised in stages (see `PLAN.md`):

- **Part 1** — dataset investigation: quality checks, anomaly detection, class balance,
  representativeness, dimensionality reduction (PCA / t-SNE / UMAP), balanced & cleaned sets.
- **Part 2** — a two-layer MLP with a grid over dataset variant × dropout × activation,
  full training/evaluation metrics.
- **Part 3** — error analysis, behaviour on saved anomalies, occlusion sensitivity maps,
  first-layer weight maps.
- **Part 4** — JAX re-implementation, framework comparison, and gradient-based sensitivity
  methods.

Key metrics and figures will be summarised here as stages complete.

## Repository structure

```
src/mnist_validation/   # package: config, data, models, training, evaluation, interpretation, visualization, cli
configs/                # YAML experiment configs (base, grid, smoke)
tests/                  # fast unit tests on synthetic arrays
notebooks/              # final executed notebook
reports/                # report.md, figures/, tables/
docs/                   # data source, representation, theory, control questions
data/                   # raw/ (not tracked), processed/, anomalies/
```

## Installation

```bash
conda env create -f environment.yml   # or reuse an existing python 3.12 env
conda activate ai-validation
pip install -e ".[dev]"
make data                             # copies the variant CSVs into data/raw/
```

## Reproduce

```bash
make lint test        # quality gate
make validate         # Part 1: data validation, figures, tables
make grid             # Part 2: training grid
make interpret        # Part 3: interpretation
make jax              # Part 4: JAX runs + comparison
make smoke            # fast end-to-end check (2 epochs, subsample)
```

Device selection is automatic for PyTorch (`mps` → `cuda` → `cpu`); JAX runs on CPU.
Hardware / timing notes will be added once the full grid has been run.

## Report and notebook

- Report: [`reports/report.md`](reports/report.md) (build `reports/report.docx` with `make report`)
- Notebook: [`notebooks/lab1_mnist_validation.ipynb`](notebooks/lab1_mnist_validation.ipynb)

## Source

Task based on: Попов И.Ю., Бучаев А.Я., Есипов Д.А. *Валидация систем искусственного
интеллекта: Лабораторный практикум.* — СПб: Университет ИТМО, 2024. — 36 с.
