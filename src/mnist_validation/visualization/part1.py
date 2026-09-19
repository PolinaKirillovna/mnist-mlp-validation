"""Figures for Part 1 (data investigation).

Every function receives already-computed data (arrays, dataframes) and writes a
PNG. Captions are in Russian for direct inclusion in the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from mnist_validation.config import IMAGE_SIDE, NUM_CLASSES
from mnist_validation.visualization.style import reason_label_ru, save_figure


def plot_class_examples(
    images: NDArray[np.uint8], labels: NDArray[np.int64], out: str | Path, per_class: int = 8
) -> Path:
    """Plot a grid of example images, one row per class.

    Args:
        images: Image array of shape ``(N, 784)``.
        labels: Class labels.
        out: Output path.
        per_class: Number of examples per class.

    Returns:
        The path written.
    """
    fig, axes = plt.subplots(
        NUM_CLASSES, per_class, figsize=(per_class, NUM_CLASSES), squeeze=False
    )
    rng = np.random.default_rng(0)
    for cls in range(NUM_CLASSES):
        cls_idx = np.flatnonzero(labels == cls)
        picks = rng.choice(cls_idx, size=min(per_class, cls_idx.size), replace=False)
        for col in range(per_class):
            ax = axes[cls][col]
            ax.axis("off")
            if col < picks.size:
                ax.imshow(images[picks[col]].reshape(IMAGE_SIDE, IMAGE_SIDE), cmap="gray")
            if col == 0:
                ax.set_ylabel(str(cls), rotation=0, labelpad=10, fontsize=11)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle("Примеры изображений по классам")
    return save_figure(fig, out)


def plot_class_histogram(labels: NDArray[np.int64], out: str | Path) -> Path:
    """Plot a bar chart of the number of objects per class.

    Args:
        labels: Class labels.
        out: Output path.

    Returns:
        The path written.
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=np.arange(NUM_CLASSES), y=counts, ax=ax, color="steelblue")
    for i, c in enumerate(counts):
        ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Класс")
    ax.set_ylabel("Количество объектов")
    ax.set_title("Распределение объектов по классам")
    return save_figure(fig, out)


def plot_balance_comparison(
    before: NDArray[np.int64], after: NDArray[np.int64], out: str | Path
) -> Path:
    """Compare class distributions before and after balancing.

    Args:
        before: Labels of the original set.
        after: Labels of the balanced set.
        out: Output path.

    Returns:
        The path written.
    """
    frame = pd.concat(
        [
            pd.DataFrame({"class": before, "Набор": "до балансировки"}),
            pd.DataFrame({"class": after, "Набор": "после балансировки"}),
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=frame, x="class", hue="Набор", ax=ax)
    ax.set_xlabel("Класс")
    ax.set_ylabel("Количество объектов")
    ax.set_title("Распределение классов до и после балансировки")
    return save_figure(fig, out)


def plot_dataset_sizes(sizes: dict[str, int], out: str | Path) -> Path:
    """Plot the sizes of the original/balanced/cleaned datasets.

    Args:
        sizes: Mapping from dataset name to sample count.
        out: Output path.

    Returns:
        The path written.
    """
    names_ru = {"original": "исходный", "balanced": "сбалансированный", "cleaned": "очищенный"}
    keys = list(sizes)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([names_ru.get(k, k) for k in keys], [sizes[k] for k in keys], color="seagreen")
    for bar, key in zip(bars, keys, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(sizes[key]),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Количество объектов")
    ax.set_title("Размеры исходного, сбалансированного и очищенного наборов")
    return save_figure(fig, out)


def plot_intensity_boxplot(stats: pd.DataFrame, column: str, out: str | Path) -> Path:
    """Plot per-class box plots of a per-image statistic.

    Args:
        stats: Dataframe from ``compute_image_statistics``.
        column: Statistic column to plot.
        out: Output path.

    Returns:
        The path written.
    """
    titles = {
        "mean_intensity": "Средняя интенсивность",
        "active_fraction": "Доля активных пикселей (заполненность)",
        "std_intensity": "Стандартное отклонение интенсивности",
        "intensity_sum": "Суммарная интенсивность",
        "nonzero_count": "Число ненулевых пикселей",
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=stats, x="label", y=column, ax=ax, color="lightsteelblue")
    ax.set_xlabel("Класс")
    ax.set_ylabel(titles.get(column, column))
    ax.set_title(f"{titles.get(column, column)} по классам")
    return save_figure(fig, out)


def plot_projection(
    coords: NDArray[np.float64],
    labels: NDArray[np.int64],
    anomaly_mask: NDArray[np.bool_],
    duplicate_mask: NDArray[np.bool_],
    title: str,
    out: str | Path,
) -> Path:
    """Plot a 2-D projection coloured by class, marking anomalies/duplicates.

    Args:
        coords: Array of shape ``(N, 2)``.
        labels: Class labels of the projected points.
        anomaly_mask: Boolean mask of anomalous points.
        duplicate_mask: Boolean mask of duplicate points.
        title: Figure title.
        out: Output path.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    normal = ~(anomaly_mask | duplicate_mask)
    scatter = ax.scatter(
        coords[normal, 0], coords[normal, 1], c=labels[normal], cmap="tab10", s=5, alpha=0.6
    )
    ax.scatter(
        coords[duplicate_mask, 0],
        coords[duplicate_mask, 1],
        marker="s",
        facecolors="none",
        edgecolors="black",
        s=28,
        linewidths=0.6,
        label="дубликаты",
    )
    ax.scatter(
        coords[anomaly_mask, 0],
        coords[anomaly_mask, 1],
        marker="x",
        c="red",
        s=32,
        linewidths=0.8,
        label="аномалии",
    )
    legend = ax.legend(*scatter.legend_elements(), title="Класс", loc="best", fontsize=7)
    ax.add_artist(legend)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Компонента 1")
    ax.set_ylabel("Компонента 2")
    return save_figure(fig, out)


def plot_examples_with_captions(
    images: NDArray[np.uint8],
    captions: list[str],
    title: str,
    out: str | Path,
    ncols: int = 6,
) -> Path:
    """Plot a grid of images with a caption under each.

    Args:
        images: Image array of shape ``(M, 784)``.
        captions: One caption per image.
        title: Figure title.
        out: Output path.
        ncols: Number of columns in the grid.

    Returns:
        The path written.
    """
    n = len(images)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.9), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].reshape(IMAGE_SIDE, IMAGE_SIDE), cmap="gray")
            ax.set_title(captions[i], fontsize=7)
    fig.suptitle(title)
    return save_figure(fig, out)


def reasons_to_caption(reasons: str) -> str:
    """Convert a ``;``-joined reason-key string into a short Russian caption."""
    keys = [r for r in reasons.split(";") if r]
    return "\n".join(reason_label_ru(k) for k in keys)
