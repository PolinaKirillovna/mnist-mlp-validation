"""Figures for Part 3 (interpretation).

Occlusion heatmaps, first-layer weight maps and activation heatmaps. Functions
receive precomputed arrays and write PNGs with Russian captions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from mnist_validation.config import IMAGE_SIDE
from mnist_validation.visualization.style import save_figure

DATASET_RU = {"original": "исходная", "balanced": "сбаланс.", "cleaned": "очищенная"}


def plot_occlusion_grid(
    image: NDArray[np.float64],
    maps: dict[str, dict[int, NDArray[np.float64]]],
    kernels: tuple[int, ...],
    title: str,
    out: str | Path,
) -> Path:
    """Plot occlusion heatmaps: rows = kernel sizes, columns = original + models.

    A single colour scale (shared across the three models and kernels of this
    image) is used so the maps are directly comparable.

    Args:
        image: Normalised image of shape ``(784,)``.
        maps: ``maps[model][kernel]`` is a 28x28 importance map.
        kernels: Kernel sizes (row order).
        title: Figure title.
        out: Output path.

    Returns:
        The path written.
    """
    models = list(maps)
    all_values = np.concatenate(
        [maps[m][k].ravel() for m in models for k in kernels if k in maps[m]]
    )
    vmax = float(np.abs(all_values).max()) or 1.0

    ncols = 1 + len(models)
    fig, axes = plt.subplots(
        len(kernels), ncols, figsize=(2.4 * ncols, 2.4 * len(kernels)), squeeze=False
    )
    img2d = image.reshape(IMAGE_SIDE, IMAGE_SIDE)
    heat = None
    for r, kernel in enumerate(kernels):
        axes[r][0].imshow(img2d, cmap="gray")
        axes[r][0].set_ylabel(f"ядро {kernel}x{kernel}", fontsize=9)
        axes[r][0].set_xticks([])
        axes[r][0].set_yticks([])
        if r == 0:
            axes[r][0].set_title("изображение", fontsize=9)
        for c, model in enumerate(models, start=1):
            ax = axes[r][c]
            heat = ax.imshow(maps[model][kernel], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.axis("off")
            if r == 0:
                ax.set_title(DATASET_RU.get(model, model), fontsize=9)
    if heat is not None:
        fig.colorbar(heat, ax=axes.ravel().tolist(), shrink=0.6, label="падение вероятности")
    fig.suptitle(title)
    return save_figure(fig, out)


def plot_weight_maps(
    maps: NDArray[np.float64], title: str, out: str | Path, n_neurons: int = 64
) -> Path:
    """Plot a grid of first-layer neuron weight maps (symmetric diverging scale).

    Args:
        maps: Array of shape ``(H, 28, 28)``.
        title: Figure title.
        out: Output path.
        n_neurons: Maximum number of neurons to display.

    Returns:
        The path written.
    """
    n = min(n_neurons, maps.shape[0])
    vmax = float(np.abs(maps[:n]).max()) or 1.0
    ncols = 8
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        if i < n:
            ax.imshow(maps[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.suptitle(title)
    return save_figure(fig, out)


def plot_activation_heatmap(heatmap: NDArray[np.float64], title: str, out: str | Path) -> Path:
    """Plot an ``(H, num_classes)`` mean-activation heatmap.

    Args:
        heatmap: Array of shape ``(H, num_classes)``.
        title: Figure title.
        out: Output path.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Нейрон первого слоя")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.6, label="средняя активация")
    return save_figure(fig, out)


def plot_example_gallery(
    images: NDArray[np.uint8],
    captions: list[str],
    title: str,
    out: str | Path,
    ncols: int = 8,
) -> Path:
    """Plot a labelled gallery of example images.

    Args:
        images: Image array of shape ``(M, 784)``.
        captions: One caption per image.
        title: Figure title.
        out: Output path.
        ncols: Columns in the grid.

    Returns:
        The path written.
    """
    n = len(images)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.8), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].reshape(IMAGE_SIDE, IMAGE_SIDE), cmap="gray")
            ax.set_title(captions[i], fontsize=7)
    fig.suptitle(title)
    return save_figure(fig, out)
