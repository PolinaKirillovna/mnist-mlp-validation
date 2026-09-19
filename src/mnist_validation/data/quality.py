"""Data-quality checks and anomaly detection.

Each check is a pure function returning a :class:`CheckResult` that records which
source rows were flagged and why. Results are combined by :func:`collect_findings`
into a single :class:`QualityReport`, from which cleaning derives the set of rows
to drop and the anomalous objects to persist for later qualitative analysis.

Reasons are stored as stable English keys (see :data:`DUPLICATE_REASONS` and
:data:`ANOMALY_REASONS`); the reporting layer maps them to Russian labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from mnist_validation.config import NUM_CLASSES, PIXEL_MAX, QualityConfig
from mnist_validation.data.loading import DatasetBundle
from mnist_validation.data.statistics import STAT_COLUMNS, compute_image_statistics

logger = logging.getLogger(__name__)

# Reason keys that denote exact/near-exact repetition (handled by de-duplication).
DUPLICATE_REASONS = ("full_duplicate", "same_label_duplicate")
# Reason keys that denote a genuinely suspicious object (persisted for Part 3).
ANOMALY_REASONS = (
    "label_conflict",
    "near_empty",
    "near_full",
    "intensity_outlier",
    "centroid_outlier",
    "knn_outlier",
    "isolation_forest_outlier",
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single quality check.

    Attributes:
        name: Stable reason key identifying the check.
        description: Human-readable English description of what was checked.
        flagged_row_ids: Source-row identifiers flagged by this check.
        detail: Optional extra information (e.g. thresholds, counts).
    """

    name: str
    description: str
    flagged_row_ids: NDArray[np.int64]
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of flagged rows."""
        return int(self.flagged_row_ids.size)


@dataclass(frozen=True)
class QualityReport:
    """Aggregated result of all quality checks over a dataset.

    Attributes:
        results: All individual check results in execution order.
        stats: Per-image statistics dataframe.
        reasons_by_row: Mapping from source row id to the set of reason keys.
    """

    results: list[CheckResult]
    stats: pd.DataFrame
    reasons_by_row: dict[int, set[str]]

    def result(self, name: str) -> CheckResult:
        """Return the check result with the given reason key."""
        for res in self.results:
            if res.name == name:
                return res
        raise KeyError(name)

    def duplicate_row_ids(self) -> NDArray[np.int64]:
        """Row ids flagged by any duplicate check (redundant copies to drop)."""
        return self._rows_for(DUPLICATE_REASONS)

    def anomaly_row_ids(self) -> NDArray[np.int64]:
        """Row ids flagged by any anomaly check (suspicious objects to isolate)."""
        return self._rows_for(ANOMALY_REASONS)

    def _rows_for(self, reasons: tuple[str, ...]) -> NDArray[np.int64]:
        rows = {row for row, found in self.reasons_by_row.items() if found.intersection(reasons)}
        return np.array(sorted(rows), dtype=np.int64)

    def summary_frame(self) -> pd.DataFrame:
        """Return a dataframe of reason key, description and flagged count."""
        return pd.DataFrame(
            {
                "reason": [r.name for r in self.results],
                "description": [r.description for r in self.results],
                "count": [r.count for r in self.results],
            }
        )


def _image_group_ids(images: NDArray[np.uint8]) -> NDArray[np.int64]:
    """Assign an integer group id to each identical image (by exact pixels)."""
    arr = np.ascontiguousarray(images)
    void_view = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    codes, _ = pd.factorize(void_view.ravel())
    return codes.astype(np.int64)


def find_full_row_duplicates(bundle: DatasetBundle) -> CheckResult:
    """Flag rows whose (label, image) pair repeats an earlier row.

    The first occurrence is kept; every later identical row is flagged.

    Args:
        bundle: Dataset to check.

    Returns:
        A :class:`CheckResult` with reason key ``full_duplicate``.
    """
    groups = _image_group_ids(bundle.images)
    frame = pd.DataFrame({"group": groups, "label": bundle.labels})
    dup_mask = frame.duplicated(keep="first").to_numpy()
    return CheckResult(
        name="full_duplicate",
        description="Exact duplicate row (identical label and pixels)",
        flagged_row_ids=bundle.row_ids[dup_mask],
    )


def find_same_label_image_duplicates(bundle: DatasetBundle) -> CheckResult:
    """Flag images that repeat within the same class (redundant copies).

    For MNIST vectors this set coincides with the full-row duplicates, but the
    check is kept separate because the two notions differ in general.

    Args:
        bundle: Dataset to check.

    Returns:
        A :class:`CheckResult` with reason key ``same_label_duplicate``.
    """
    groups = _image_group_ids(bundle.images)
    frame = pd.DataFrame({"group": groups, "label": bundle.labels})
    dup_mask = frame.duplicated(subset=["group", "label"], keep="first").to_numpy()
    return CheckResult(
        name="same_label_duplicate",
        description="Duplicate image within the same class",
        flagged_row_ids=bundle.row_ids[dup_mask],
    )


def find_label_conflicts(bundle: DatasetBundle) -> CheckResult:
    """Flag identical images that appear with more than one label.

    Args:
        bundle: Dataset to check.

    Returns:
        A :class:`CheckResult` with reason key ``label_conflict``.
    """
    groups = _image_group_ids(bundle.images)
    frame = pd.DataFrame({"group": groups, "label": bundle.labels})
    distinct_labels = frame.groupby("group")["label"].nunique()
    conflict_groups = distinct_labels[distinct_labels > 1].index
    mask = frame["group"].isin(conflict_groups).to_numpy()
    return CheckResult(
        name="label_conflict",
        description="Identical image assigned conflicting labels",
        flagged_row_ids=bundle.row_ids[mask],
    )


def find_out_of_range(bundle: DatasetBundle, pixel_max: int = PIXEL_MAX) -> CheckResult:
    """Flag rows containing pixel values outside ``[0, pixel_max]``.

    Args:
        bundle: Dataset to check.
        pixel_max: Maximum valid pixel value.

    Returns:
        A :class:`CheckResult` with reason key ``out_of_range``.
    """
    bad = (bundle.images < 0) | (bundle.images > pixel_max)
    mask = bad.any(axis=1)
    return CheckResult(
        name="out_of_range",
        description=f"Pixel value outside [0, {pixel_max}]",
        flagged_row_ids=bundle.row_ids[mask],
    )


def find_empty_or_full(bundle: DatasetBundle, config: QualityConfig) -> list[CheckResult]:
    """Flag near-empty and near-completely-filled images.

    Thresholds come from the config and are justified in the report: an MNIST
    digit typically activates 10-25% of the pixels, so images below
    ``empty_active_fraction`` carry almost no signal and images above
    ``filled_active_fraction`` are effectively saturated.

    Args:
        bundle: Dataset to check.
        config: Quality thresholds.

    Returns:
        A list with the ``near_empty`` and ``near_full`` check results.
    """
    stats = compute_image_statistics(
        bundle.images, bundle.labels, bundle.row_ids, config.active_pixel_threshold
    )
    active = stats["active_fraction"].to_numpy()
    empty_mask = active <= config.empty_active_fraction
    full_mask = active >= config.filled_active_fraction
    return [
        CheckResult(
            name="near_empty",
            description="Near-empty image (too few active pixels)",
            flagged_row_ids=bundle.row_ids[empty_mask],
            detail={"threshold": config.empty_active_fraction},
        ),
        CheckResult(
            name="near_full",
            description="Near-completely-filled image (too many active pixels)",
            flagged_row_ids=bundle.row_ids[full_mask],
            detail={"threshold": config.filled_active_fraction},
        ),
    ]


def _within_class_zscore(
    values: NDArray[np.float64], labels: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return absolute z-scores computed within each class."""
    z = np.zeros_like(values, dtype=np.float64)
    for cls in np.unique(labels):
        mask = labels == cls
        col = values[mask]
        std = col.std()
        if std > 0:
            z[mask] = np.abs((col - col.mean()) / std)
    return z


def find_intensity_outliers(bundle: DatasetBundle, config: QualityConfig) -> CheckResult:
    """Flag images whose intensity statistics are outliers within their class.

    A row is flagged if the absolute within-class z-score of *any* statistic in
    :data:`~mnist_validation.data.statistics.STAT_COLUMNS` exceeds the configured
    threshold.

    Args:
        bundle: Dataset to check.
        config: Quality thresholds.

    Returns:
        A :class:`CheckResult` with reason key ``intensity_outlier``.
    """
    stats = compute_image_statistics(
        bundle.images, bundle.labels, bundle.row_ids, config.active_pixel_threshold
    )
    flagged = np.zeros(len(bundle), dtype=bool)
    for column in STAT_COLUMNS:
        z = _within_class_zscore(stats[column].to_numpy(dtype=np.float64), bundle.labels)
        flagged |= z > config.intensity_z_threshold
    return CheckResult(
        name="intensity_outlier",
        description="Outlier intensity/fill statistic within its class",
        flagged_row_ids=bundle.row_ids[flagged],
        detail={"z_threshold": config.intensity_z_threshold},
    )


def find_centroid_outliers(bundle: DatasetBundle, config: QualityConfig) -> CheckResult:
    """Flag images far from their class centroid in pixel space.

    Distance is the Euclidean distance to the per-class mean image; a row is
    flagged when its within-class z-score of that distance exceeds the threshold.

    Args:
        bundle: Dataset to check.
        config: Quality thresholds.

    Returns:
        A :class:`CheckResult` with reason key ``centroid_outlier``.
    """
    pixels = bundle.images.astype(np.float64)
    distances = np.zeros(len(bundle), dtype=np.float64)
    for cls in range(NUM_CLASSES):
        mask = bundle.labels == cls
        if not mask.any():
            continue
        centroid = pixels[mask].mean(axis=0)
        distances[mask] = np.linalg.norm(pixels[mask] - centroid, axis=1)
    z = _within_class_zscore(distances, bundle.labels)
    mask = z > config.centroid_z_threshold
    return CheckResult(
        name="centroid_outlier",
        description="Far from class centroid in pixel space",
        flagged_row_ids=bundle.row_ids[mask],
        detail={"z_threshold": config.centroid_z_threshold},
    )


def _pca_features(images: NDArray[np.uint8], n_components: int, seed: int) -> NDArray[np.float64]:
    """Project images to ``n_components`` PCA dimensions (scaled to [0, 1])."""
    pixels = images.astype(np.float64) / PIXEL_MAX
    n_components = min(n_components, pixels.shape[1], pixels.shape[0])
    pca = PCA(n_components=n_components, random_state=seed)
    return cast("NDArray[np.float64]", pca.fit_transform(pixels).astype(np.float64))


def find_knn_outliers(
    bundle: DatasetBundle, config: QualityConfig, seed: int, pca_components: int = 50
) -> CheckResult:
    """Flag images with a large mean distance to their k nearest neighbours.

    The distance is computed in a PCA-reduced space (global density estimate);
    rows above the configured distance quantile are flagged.

    Args:
        bundle: Dataset to check.
        config: Quality thresholds.
        seed: Random seed for PCA.
        pca_components: Number of PCA components for the neighbour search.

    Returns:
        A :class:`CheckResult` with reason key ``knn_outlier``.
    """
    features = _pca_features(bundle.images, pca_components, seed)
    k = min(config.knn_neighbors, len(bundle) - 1)
    if k < 1:
        return CheckResult(
            "knn_outlier", "kNN-distance outlier in PCA space", np.array([], dtype=np.int64)
        )
    nn = NearestNeighbors(n_neighbors=k + 1).fit(features)
    distances, _ = nn.kneighbors(features)
    mean_dist = distances[:, 1:].mean(axis=1)
    cutoff = float(np.quantile(mean_dist, config.knn_quantile))
    mask = mean_dist > cutoff
    return CheckResult(
        name="knn_outlier",
        description="kNN-distance outlier in PCA space",
        flagged_row_ids=bundle.row_ids[mask],
        detail={"quantile": config.knn_quantile, "cutoff": cutoff},
    )


def find_isolation_forest_outliers(
    bundle: DatasetBundle, config: QualityConfig, seed: int, pca_components: int = 50
) -> CheckResult:
    """Flag images that an Isolation Forest isolates as outliers in PCA space.

    Args:
        bundle: Dataset to check.
        config: Quality thresholds.
        seed: Random seed for PCA and the forest.
        pca_components: Number of PCA components used as features.

    Returns:
        A :class:`CheckResult` with reason key ``isolation_forest_outlier``.
    """
    features = _pca_features(bundle.images, pca_components, seed)
    forest = IsolationForest(
        contamination=config.isolation_contamination, random_state=seed, n_jobs=-1
    )
    predictions = forest.fit_predict(features)
    mask = predictions == -1
    return CheckResult(
        name="isolation_forest_outlier",
        description="Isolation Forest outlier in PCA space",
        flagged_row_ids=bundle.row_ids[mask],
        detail={"contamination": config.isolation_contamination},
    )


def run_all_checks(
    bundle: DatasetBundle, config: QualityConfig, seed: int, pca_components: int = 50
) -> list[CheckResult]:
    """Run every quality check on a dataset.

    Args:
        bundle: Dataset to check.
        config: Quality thresholds.
        seed: Random seed for stochastic detectors.
        pca_components: PCA dimensionality for projection-space detectors.

    Returns:
        The list of all check results.
    """
    logger.info("Running quality checks on '%s' (%d samples)", bundle.name, len(bundle))
    results: list[CheckResult] = [
        find_full_row_duplicates(bundle),
        find_same_label_image_duplicates(bundle),
        find_label_conflicts(bundle),
        find_out_of_range(bundle),
        *find_empty_or_full(bundle, config),
        find_intensity_outliers(bundle, config),
        find_centroid_outliers(bundle, config),
        find_knn_outliers(bundle, config, seed, pca_components),
        find_isolation_forest_outliers(bundle, config, seed, pca_components),
    ]
    for res in results:
        logger.info("  %-26s %6d flagged", res.name, res.count)
    return results


def collect_findings(
    bundle: DatasetBundle, config: QualityConfig, seed: int, pca_components: int = 50
) -> QualityReport:
    """Run all checks and aggregate them into a :class:`QualityReport`.

    Args:
        bundle: Dataset to check.
        config: Quality thresholds.
        seed: Random seed for stochastic detectors.
        pca_components: PCA dimensionality for projection-space detectors.

    Returns:
        The aggregated quality report.
    """
    results = run_all_checks(bundle, config, seed, pca_components)
    reasons_by_row: dict[int, set[str]] = {}
    for res in results:
        for row_id in res.flagged_row_ids.tolist():
            reasons_by_row.setdefault(int(row_id), set()).add(res.name)
    stats = compute_image_statistics(
        bundle.images, bundle.labels, bundle.row_ids, config.active_pixel_threshold
    )
    return QualityReport(results=results, stats=stats, reasons_by_row=reasons_by_row)
