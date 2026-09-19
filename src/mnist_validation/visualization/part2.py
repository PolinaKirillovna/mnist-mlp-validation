"""Figures for Part 2 (training and model comparison).

Functions receive already-computed histories/metrics and write PNGs with Russian
captions for the report.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from mnist_validation.config import NUM_CLASSES
from mnist_validation.visualization.style import save_figure

DATASET_RU = {"original": "исходный", "balanced": "сбалансированный", "cleaned": "очищенный"}


def plot_final_learning_curves(
    history: list[dict[str, float]], title: str, out: str | Path
) -> Path:
    """Plot train/val loss, accuracy and macro-F1 curves for one model.

    Args:
        history: Per-epoch metrics.
        title: Figure title.
        out: Output path.

    Returns:
        The path written.
    """
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [
        ("loss", "train_loss", "val_loss", "Функция потерь"),
        ("accuracy", "train_accuracy", "val_accuracy", "Точность"),
        ("f1", "train_macro_f1", "val_macro_f1", "Macro-F1"),
    ]
    for ax, (_, train_key, val_key, ylabel) in zip(axes, specs, strict=True):
        ax.plot(frame["epoch"], frame[train_key], label="обучение", marker="o", ms=3)
        ax.plot(frame["epoch"], frame[val_key], label="валидация", marker="s", ms=3)
        ax.set_xlabel("Эпоха")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    return save_figure(fig, out)


def plot_grid_learning_curves(
    records: list[dict[str, object]],
    datasets: tuple[str, ...],
    activations: tuple[str, ...],
    out: str | Path,
) -> Path:
    """Plot a small-multiples grid of validation macro-F1 curves for all runs.

    Rows are dataset variants, columns are activations; each panel overlays the
    dropout values.

    Args:
        records: Run records with keys ``dataset``, ``activation``, ``dropout``,
            ``history``.
        datasets: Dataset variants (row order).
        activations: Activations (column order).
        out: Output path.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(
        len(datasets),
        len(activations),
        figsize=(4 * len(activations), 3 * len(datasets)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for r, dataset in enumerate(datasets):
        for c, activation in enumerate(activations):
            ax = axes[r][c]
            for rec in records:
                if rec["dataset"] == dataset and rec["activation"] == activation:
                    frame = pd.DataFrame(cast("list[dict[str, float]]", rec["history"]))
                    ax.plot(frame["epoch"], frame["val_macro_f1"], label=f"p={rec['dropout']}")
            if r == 0:
                ax.set_title(activation)
            if c == 0:
                ax.set_ylabel(f"{DATASET_RU.get(dataset, dataset)}\nval macro-F1")
            if r == len(datasets) - 1:
                ax.set_xlabel("Эпоха")
            ax.legend(fontsize=7)
    fig.suptitle("Кривые обучения (валидационный macro-F1) для всех прогонов")
    return save_figure(fig, out)


def plot_confusion_matrix(conf: NDArray[np.int64], title: str, out: str | Path) -> Path:
    """Plot a 10x10 confusion matrix heatmap.

    Args:
        conf: Confusion matrix of shape ``(10, 10)``.
        title: Figure title.
        out: Output path.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        conf,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=np.arange(NUM_CLASSES),
        yticklabels=np.arange(NUM_CLASSES),
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    return save_figure(fig, out)


def plot_metric_bars(summary: pd.DataFrame, out: str | Path) -> Path:
    """Plot grouped bars of test accuracy / macro-F1 / weighted-F1 by dataset.

    Args:
        summary: Dataframe with columns ``dataset`` and the metric columns.
        out: Output path.

    Returns:
        The path written.
    """
    metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]
    labels_ru = {
        "test_accuracy": "accuracy",
        "test_macro_f1": "macro-F1",
        "test_weighted_f1": "weighted-F1",
    }
    melted = summary.melt(
        id_vars="dataset", value_vars=metrics, var_name="Метрика", value_name="Значение"
    )
    melted["Метрика"] = melted["Метрика"].map(labels_ru)
    melted["dataset"] = melted["dataset"].map(lambda d: DATASET_RU.get(d, d))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=melted, x="dataset", y="Значение", hue="Метрика", ax=ax)
    ax.set_ylim(0.9, 1.0)
    ax.set_xlabel("Набор данных")
    ax.set_title("Метрики итоговых моделей на тестовой выборке")
    return save_figure(fig, out)


def plot_hyperparameter_effect(grid: pd.DataFrame, out: str | Path) -> Path:
    """Plot the effect of dropout and activation on validation macro-F1.

    Args:
        grid: Full grid dataframe with columns ``dataset``, ``activation``,
            ``dropout`` and ``val_macro_f1``.
        out: Output path.

    Returns:
        The path written.
    """
    datasets = list(grid["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets, strict=True):
        sub = grid[grid["dataset"] == dataset]
        for activation in sub["activation"].unique():
            line = sub[sub["activation"] == activation].sort_values("dropout")
            ax.plot(line["dropout"], line["val_macro_f1"], marker="o", label=activation)
        ax.set_title(DATASET_RU.get(dataset, dataset))
        ax.set_xlabel("dropout p")
        ax.set_ylabel("val macro-F1")
        ax.legend(fontsize=8)
    fig.suptitle("Влияние dropout и функции активации на валидационный macro-F1")
    return save_figure(fig, out)
