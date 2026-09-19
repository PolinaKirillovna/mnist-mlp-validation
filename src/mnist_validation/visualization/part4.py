"""Figures for Part 4 (PyTorch vs JAX comparison and gradient methods)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from mnist_validation.config import IMAGE_SIDE
from mnist_validation.visualization.style import save_figure

DATASET_RU = {"original": "исходная", "balanced": "сбаланс.", "cleaned": "очищенная"}
METHOD_RU = {
    "saliency": "Saliency",
    "input_x_gradient": "Input x Gradient",
    "integrated_gradients": "Integrated Gradients",
    "occlusion": "Окклюзия",
}


def plot_framework_metric_bars(comparison: pd.DataFrame, out: str | Path) -> Path:
    """Plot test macro-F1 by dataset and framework.

    Args:
        comparison: Dataframe with ``dataset``, ``framework`` and ``test_macro_f1``.
        out: Output path.

    Returns:
        The path written.
    """
    frame = comparison.copy()
    frame["dataset"] = frame["dataset"].map(lambda d: DATASET_RU.get(d, d))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=frame, x="dataset", y="test_macro_f1", hue="framework", ax=ax)
    ax.set_ylim(0.85, 1.0)
    ax.set_xlabel("Набор данных")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Сравнение PyTorch и JAX по итоговому macro-F1")
    return save_figure(fig, out)


def plot_stability(stability: pd.DataFrame, out: str | Path) -> Path:
    """Plot mean +/- std of test macro-F1 over seeds, by dataset and framework.

    Args:
        stability: Dataframe with ``dataset``, ``framework``, ``mean`` and ``std``.
        out: Output path.

    Returns:
        The path written.
    """
    frame = stability.copy()
    frame["dataset"] = frame["dataset"].map(lambda d: DATASET_RU.get(d, d))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    datasets = list(frame["dataset"].unique())
    frameworks = list(frame["framework"].unique())
    width = 0.35
    x = np.arange(len(datasets))
    for i, fw in enumerate(frameworks):
        sub = frame[frame["framework"] == fw].set_index("dataset").reindex(datasets)
        ax.bar(x + (i - 0.5) * width, sub["mean"], width, yerr=sub["std"], capsize=4, label=fw)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("test macro-F1 (среднее +/- ст.откл., 3 запуска)")
    ax.set_title("Стабильность результатов по трём запускам")
    ax.legend()
    return save_figure(fig, out)


def plot_lr_overlay(
    torch_history: list[dict[str, float]],
    jax_history: list[dict[str, float]],
    title: str,
    out: str | Path,
) -> Path:
    """Overlay PyTorch and JAX validation macro-F1 learning curves.

    Args:
        torch_history: PyTorch per-epoch history.
        jax_history: JAX per-epoch history.
        title: Figure title.
        out: Output path.

    Returns:
        The path written.
    """
    tf = pd.DataFrame(torch_history)
    jf = pd.DataFrame(jax_history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tf["epoch"], tf["val_macro_f1"], marker="o", ms=3, label="PyTorch")
    ax.plot(jf["epoch"], jf["val_macro_f1"], marker="s", ms=3, label="JAX")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("val macro-F1")
    ax.set_title(title)
    ax.legend()
    return save_figure(fig, out)


def plot_gradient_methods(
    image: NDArray[np.float64],
    maps: dict[str, dict[str, NDArray[np.float64]]],
    methods: tuple[str, ...],
    title: str,
    out: str | Path,
) -> Path:
    """Plot gradient attribution maps: rows = methods, columns = image + frameworks.

    Args:
        image: Normalised image of shape ``(784,)``.
        maps: ``maps[framework][method]`` is a 784-length attribution.
        methods: Method keys (row order).
        title: Figure title.
        out: Output path.

    Returns:
        The path written.
    """
    frameworks = list(maps)
    ncols = 1 + len(frameworks)
    fig, axes = plt.subplots(
        len(methods), ncols, figsize=(2.4 * ncols, 2.4 * len(methods)), squeeze=False
    )
    img2d = image.reshape(IMAGE_SIDE, IMAGE_SIDE)
    for r, method in enumerate(methods):
        vals = np.concatenate([maps[f][method].ravel() for f in frameworks])
        vmax = float(np.abs(vals).max()) or 1.0
        axes[r][0].imshow(img2d, cmap="gray")
        axes[r][0].set_ylabel(METHOD_RU.get(method, method), fontsize=8)
        axes[r][0].set_xticks([])
        axes[r][0].set_yticks([])
        if r == 0:
            axes[r][0].set_title("изображение", fontsize=9)
        for c, fw in enumerate(frameworks, start=1):
            ax = axes[r][c]
            ax.imshow(
                maps[fw][method].reshape(IMAGE_SIDE, IMAGE_SIDE),
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.axis("off")
            if r == 0:
                ax.set_title(fw, fontsize=9)
    fig.suptitle(title)
    return save_figure(fig, out)
