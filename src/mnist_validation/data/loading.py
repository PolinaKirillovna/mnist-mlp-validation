"""Dataset loading and format validation.

Reads a variant CSV (``label`` + ``1x1`` … ``28x28``) into a
:class:`DatasetBundle` holding ``uint8`` images of shape ``(N, 784)``, integer
labels and stable row identifiers. Loading also enforces the expected format
(column count, value range, absence of NaN) and fails loudly otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mnist_validation.config import NUM_PIXELS, PIXEL_MAX

logger = logging.getLogger(__name__)

LABEL_COLUMN = "label"


@dataclass(frozen=True)
class DatasetBundle:
    """An immutable image dataset in flattened vector form.

    Attributes:
        images: ``uint8`` array of shape ``(N, 784)`` with values in ``[0, 255]``.
        labels: ``int64`` array of shape ``(N,)`` with class labels in ``[0, 9]``.
        row_ids: ``int64`` array of shape ``(N,)`` with stable source-row indices
            (0-based position in the original CSV, excluding the header).
        name: Human-readable name of the dataset (e.g. ``"train"``, ``"test"``).
    """

    images: NDArray[np.uint8]
    labels: NDArray[np.int64]
    row_ids: NDArray[np.int64]
    name: str

    def __post_init__(self) -> None:
        """Validate array shapes and dtypes on construction."""
        n = self.images.shape[0]
        if self.images.ndim != 2 or self.images.shape[1] != NUM_PIXELS:
            raise ValueError(f"images must have shape (N, {NUM_PIXELS}), got {self.images.shape}")
        if self.labels.shape != (n,) or self.row_ids.shape != (n,):
            raise ValueError("labels and row_ids must have shape (N,)")

    def __len__(self) -> int:
        """Number of samples in the bundle."""
        return int(self.images.shape[0])

    @property
    def num_samples(self) -> int:
        """Number of samples in the bundle."""
        return len(self)

    def images_2d(self) -> NDArray[np.uint8]:
        """Return images reshaped to ``(N, 28, 28)`` without copying data twice."""
        side = int(round(NUM_PIXELS**0.5))
        return self.images.reshape(-1, side, side)

    def subset(self, indices: NDArray[np.integer[Any]], name: str | None = None) -> DatasetBundle:
        """Return a new bundle restricted to ``indices``.

        Args:
            indices: Integer index array selecting rows.
            name: Optional new name; defaults to the current name.

        Returns:
            A new :class:`DatasetBundle` sharing no state with the original.
        """
        idx = np.asarray(indices, dtype=np.int64)
        return DatasetBundle(
            images=self.images[idx].copy(),
            labels=self.labels[idx].copy(),
            row_ids=self.row_ids[idx].copy(),
            name=name or self.name,
        )


def _pixel_columns(columns: list[str]) -> list[str]:
    """Return the pixel column names (everything except the label column)."""
    return [c for c in columns if c != LABEL_COLUMN]


def load_bundle(path: str | Path, name: str) -> DatasetBundle:
    """Load and validate a variant CSV into a :class:`DatasetBundle`.

    Args:
        path: Path to the CSV file.
        name: Name to attach to the resulting bundle.

    Returns:
        The validated :class:`DatasetBundle`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the format (columns, range, NaN) is unexpected.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. Run `make data` to populate data/raw/."
        )

    logger.info("Loading dataset '%s' from %s", name, csv_path)
    frame = pd.read_csv(csv_path)
    validate_frame(frame, name)

    pixel_cols = _pixel_columns(frame.columns.tolist())
    images = frame[pixel_cols].to_numpy(dtype=np.uint8)
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int64)
    row_ids = np.arange(len(frame), dtype=np.int64)

    logger.info("Loaded '%s': %d samples, %d features", name, len(labels), images.shape[1])
    return DatasetBundle(images=images, labels=labels, row_ids=row_ids, name=name)


def save_bundle(bundle: DatasetBundle, path: str | Path) -> Path:
    """Persist a bundle to a compressed ``.npz`` file.

    Args:
        bundle: Bundle to save.
        path: Output ``.npz`` path.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=bundle.images,
        labels=bundle.labels,
        row_ids=bundle.row_ids,
        name=np.array(bundle.name),
    )
    return out


def load_bundle_npz(path: str | Path) -> DatasetBundle:
    """Load a bundle previously saved with :func:`save_bundle`.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        The reconstructed :class:`DatasetBundle`.
    """
    data = np.load(path, allow_pickle=False)
    return DatasetBundle(
        images=data["images"].astype(np.uint8),
        labels=data["labels"].astype(np.int64),
        row_ids=data["row_ids"].astype(np.int64),
        name=str(data["name"]),
    )


def validate_frame(frame: pd.DataFrame, name: str) -> None:
    """Validate the raw dataframe format and fail with a clear message otherwise.

    Args:
        frame: Raw dataframe as read from CSV.
        name: Dataset name for error messages.

    Raises:
        ValueError: If any format expectation is violated.
    """
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"[{name}] missing '{LABEL_COLUMN}' column")

    pixel_cols = _pixel_columns(frame.columns.tolist())
    if len(pixel_cols) != NUM_PIXELS:
        raise ValueError(f"[{name}] expected {NUM_PIXELS} pixel columns, got {len(pixel_cols)}")

    if frame.isna().to_numpy().any():
        raise ValueError(f"[{name}] contains NaN values")

    pixels = frame[pixel_cols].to_numpy()
    px_min, px_max = int(pixels.min()), int(pixels.max())
    if px_min < 0 or px_max > PIXEL_MAX:
        raise ValueError(f"[{name}] pixel values out of range: [{px_min}, {px_max}]")

    labels = frame[LABEL_COLUMN].to_numpy()
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError(f"[{name}] labels out of range: [{labels.min()}, {labels.max()}]")
