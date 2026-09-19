"""Shared plotting style and helpers.

Uses a non-interactive backend so figures render headless (CI, scripts). Figures
are saved at a fixed DPI with Russian captions, since they are embedded in the
Russian-language report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

FIGURE_DPI = 200

# Stable Russian labels for anomaly/duplicate reason keys used across figures/tables.
REASON_LABELS_RU: dict[str, str] = {
    "full_duplicate": "полный дубликат строки",
    "same_label_duplicate": "дубликат изображения с той же меткой",
    "label_conflict": "конфликт меток",
    "out_of_range": "значение вне диапазона",
    "near_empty": "почти пустое изображение",
    "near_full": "почти полностью заполненное",
    "intensity_outlier": "выброс по статистике интенсивности",
    "centroid_outlier": "далеко от центроида класса",
    "knn_outlier": "выброс по kNN-расстоянию",
    "isolation_forest_outlier": "выброс (Isolation Forest)",
}


def apply_style() -> None:
    """Apply the shared seaborn/matplotlib style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = FIGURE_DPI
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["font.size"] = 10


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save a figure at the report DPI and close it.

    Args:
        fig: Matplotlib figure to save.
        path: Destination path (parent directories are created).

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def reason_label_ru(reason_key: str) -> str:
    """Return the Russian label for a reason key (falling back to the key)."""
    return REASON_LABELS_RU.get(reason_key, reason_key)
