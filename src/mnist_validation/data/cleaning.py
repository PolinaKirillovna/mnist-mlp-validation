"""Dataset cleaning and balancing.

Builds the three training variants required by the assignment:

* ``original`` — the training split untouched;
* ``balanced`` — undersampled so every class has the size of the smallest class;
* ``cleaned`` — the balanced set with duplicates and anomalies removed first.

Undersampling is chosen for balancing because class 9 is the binding constraint
(only ~2k objects) and oversampling would duplicate an already small class; this
trade-off is discussed in the report. Anomalies detected on the full training
variant are persisted for the qualitative analysis in Part 3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mnist_validation.config import NUM_CLASSES
from mnist_validation.data.loading import DatasetBundle
from mnist_validation.data.quality import QualityReport

logger = logging.getLogger(__name__)


def undersample_to_min(bundle: DatasetBundle, seed: int) -> DatasetBundle:
    """Undersample every class to the size of the smallest present class.

    Args:
        bundle: Dataset to balance.
        seed: Random seed for reproducible sampling.

    Returns:
        A class-balanced :class:`DatasetBundle`.
    """
    rng = np.random.default_rng(seed)
    present = [c for c in range(NUM_CLASSES) if np.any(bundle.labels == c)]
    min_count = min(int(np.sum(bundle.labels == c)) for c in present)
    selected: list[NDArray[np.int64]] = []
    for cls in present:
        cls_idx = np.flatnonzero(bundle.labels == cls)
        chosen = rng.choice(cls_idx, size=min_count, replace=False)
        selected.append(chosen)
    idx = np.sort(np.concatenate(selected))
    logger.info("Balanced '%s' to %d per class (%d total)", bundle.name, min_count, idx.size)
    return bundle.subset(idx, name="balanced")


def remove_row_ids(bundle: DatasetBundle, row_ids: NDArray[np.int64], name: str) -> DatasetBundle:
    """Return a bundle with the given source row ids removed.

    Args:
        bundle: Dataset to filter.
        row_ids: Source row ids to drop.
        name: Name for the resulting bundle.

    Returns:
        A filtered :class:`DatasetBundle`.
    """
    drop = set(row_ids.tolist())
    keep_mask = np.array([rid not in drop for rid in bundle.row_ids.tolist()], dtype=bool)
    return bundle.subset(np.flatnonzero(keep_mask), name=name)


def build_variants(
    train_raw: DatasetBundle, report: QualityReport, seed: int
) -> dict[str, DatasetBundle]:
    """Build the ``original``, ``balanced`` and ``cleaned`` training variants.

    Args:
        train_raw: The raw training split (before balancing/cleaning).
        report: Quality report used to identify duplicates and anomalies.
        seed: Random seed for reproducible undersampling.

    Returns:
        A mapping with keys ``original``, ``balanced`` and ``cleaned``.
    """
    original = train_raw.subset(np.arange(len(train_raw)), name="original")

    balanced = undersample_to_min(train_raw, seed)

    to_drop = np.union1d(report.duplicate_row_ids(), report.anomaly_row_ids())
    filtered = remove_row_ids(train_raw, to_drop, name="filtered")
    cleaned = undersample_to_min(filtered, seed)
    cleaned = DatasetBundle(
        images=cleaned.images, labels=cleaned.labels, row_ids=cleaned.row_ids, name="cleaned"
    )

    logger.info(
        "Variants: original=%d, balanced=%d, cleaned=%d",
        len(original),
        len(balanced),
        len(cleaned),
    )
    return {"original": original, "balanced": balanced, "cleaned": cleaned}


def anomalies_frame(bundle: DatasetBundle, report: QualityReport) -> pd.DataFrame:
    """Build a dataframe of anomalous objects with their reasons and pixels.

    Args:
        bundle: The dataset the report was computed on (full training variant).
        report: Quality report holding the per-row reasons.

    Returns:
        A dataframe with ``row_id``, ``label``, ``reasons`` and ``px0``..``px783``.
    """
    anomaly_ids = report.anomaly_row_ids()
    positions = {int(rid): i for i, rid in enumerate(bundle.row_ids.tolist())}
    rows = [positions[int(rid)] for rid in anomaly_ids.tolist()]
    if not rows:
        columns = ["row_id", "label", "reasons", *[f"px{i}" for i in range(bundle.images.shape[1])]]
        return pd.DataFrame(columns=columns)

    images = bundle.images[rows]
    labels = bundle.labels[rows]
    reasons = [";".join(sorted(report.reasons_by_row[int(rid)])) for rid in anomaly_ids.tolist()]
    meta = pd.DataFrame({"row_id": anomaly_ids, "label": labels, "reasons": reasons})
    pixels = pd.DataFrame(images, columns=[f"px{i}" for i in range(images.shape[1])])
    return pd.concat([meta, pixels], axis=1)


def save_anomalies(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist the anomalies dataframe to parquet.

    Args:
        frame: Anomalies dataframe from :func:`anomalies_frame`.
        path: Output parquet path.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    logger.info("Saved %d anomalies to %s", len(frame), out)
    return out


def load_anomalies(path: str | Path) -> tuple[DatasetBundle, pd.DataFrame]:
    """Load persisted anomalies back into a bundle plus a metadata frame.

    Args:
        path: Parquet path written by :func:`save_anomalies`.

    Returns:
        A ``(bundle, meta)`` tuple where ``meta`` holds ``row_id``, ``label`` and
        ``reasons`` columns.
    """
    frame = pd.read_parquet(path)
    pixel_cols = [c for c in frame.columns if c.startswith("px")]
    images = frame[pixel_cols].to_numpy(dtype=np.uint8)
    labels = frame["label"].to_numpy(dtype=np.int64)
    row_ids = frame["row_id"].to_numpy(dtype=np.int64)
    bundle = DatasetBundle(images=images, labels=labels, row_ids=row_ids, name="anomalies")
    return bundle, frame[["row_id", "label", "reasons"]]
