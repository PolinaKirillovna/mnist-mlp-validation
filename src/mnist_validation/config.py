"""Experiment configuration.

Frozen dataclasses describing every tunable parameter of the pipeline, plus a
loader that reads a YAML file into a fully typed :class:`Config`. Nothing in the
package reads magic numbers directly; they all flow through this module so that
experiments are reproducible and self-documenting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, cast

import yaml

# Domain constants that are fixed by the MNIST problem itself rather than tunable.
IMAGE_SIDE: int = 28
NUM_PIXELS: int = IMAGE_SIDE * IMAGE_SIDE
NUM_CLASSES: int = 10
PIXEL_MAX: int = 255


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used across the pipeline.

    Attributes:
        raw_dir: Directory holding the raw variant CSV files (not tracked in git).
        train_file: Training CSV file name inside ``raw_dir``.
        test_file: Shared test CSV file name inside ``raw_dir``.
        processed_dir: Output directory for processed datasets.
        anomalies_dir: Output directory for saved anomalous objects.
        artifacts_dir: Output directory for checkpoints and metrics.
        figures_dir: Output directory for report figures.
        tables_dir: Output directory for report tables.
    """

    raw_dir: Path = Path("data/raw")
    train_file: str = "d3.csv"
    test_file: str = "mnist_test.csv"
    processed_dir: Path = Path("data/processed")
    anomalies_dir: Path = Path("data/anomalies")
    artifacts_dir: Path = Path("artifacts")
    figures_dir: Path = Path("reports/figures")
    tables_dir: Path = Path("reports/tables")

    @property
    def train_path(self) -> Path:
        """Full path to the training CSV."""
        return self.raw_dir / self.train_file

    @property
    def test_path(self) -> Path:
        """Full path to the test CSV."""
        return self.raw_dir / self.test_file


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for data-quality checks and outlier detection.

    Attributes:
        empty_active_fraction: Fraction of active pixels below which an image is
            considered (near-)empty.
        filled_active_fraction: Fraction of active pixels above which an image is
            considered (near-)completely filled.
        active_pixel_threshold: Pixel value (on the 0-255 scale) above which a
            pixel counts as "active".
        intensity_z_threshold: Absolute z-score threshold for per-class intensity
            statistic outliers.
        iqr_multiplier: Multiplier for the inter-quartile-range outlier rule.
        centroid_z_threshold: Absolute z-score threshold for distance to the
            per-class centroid.
        knn_neighbors: Number of neighbours for the k-NN distance outlier score
            in the projected space.
        knn_quantile: Upper quantile of k-NN distances flagged as outliers.
        isolation_contamination: Expected outlier fraction for Isolation Forest.
    """

    empty_active_fraction: float = 0.02
    filled_active_fraction: float = 0.95
    active_pixel_threshold: int = 0
    intensity_z_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    centroid_z_threshold: float = 3.0
    knn_neighbors: int = 20
    knn_quantile: float = 0.995
    isolation_contamination: float = 0.01


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation split policy (the test set is supplied separately).

    Attributes:
        val_fraction: Fraction of the training variant held out for validation.
        stratified: Whether the split preserves per-class proportions.
    """

    val_fraction: float = 0.1
    stratified: bool = True


@dataclass(frozen=True)
class DimReductionConfig:
    """Dimensionality-reduction settings for visualisation.

    Attributes:
        subsample_size: Number of stratified samples used for the projections.
        pca_components: Number of PCA components computed.
        tsne_perplexity: t-SNE perplexity.
        umap_neighbors: UMAP ``n_neighbors``.
        umap_min_dist: UMAP ``min_dist``.
    """

    subsample_size: int = 10_000
    pca_components: int = 50
    tsne_perplexity: float = 30.0
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    """Two-layer perceptron architecture.

    Attributes:
        hidden_sizes: Sizes of the hidden layers (two trainable hidden layers).
        activation: Activation function name (``relu``, ``tanh`` or ``gelu``).
        dropout_p: Dropout probability applied after each hidden activation.
    """

    hidden_sizes: tuple[int, ...] = (256, 128)
    activation: str = "relu"
    dropout_p: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation and early-stopping settings.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        early_stopping_patience: Epochs without val macro-F1 improvement before
            stopping.
        num_workers: DataLoader worker processes (PyTorch).
    """

    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    num_workers: int = 0


@dataclass(frozen=True)
class GridConfig:
    """Experiment grid over datasets, dropout values and activations.

    Attributes:
        datasets: Dataset variants to sweep (``original``, ``balanced``, ``cleaned``).
        dropouts: Dropout probabilities to sweep.
        activations: Activation functions to sweep.
        stability_repeats: Number of repeated runs used for stability analysis.
    """

    datasets: tuple[str, ...] = ("original", "balanced", "cleaned")
    dropouts: tuple[float, ...] = (0.0, 0.2, 0.5)
    activations: tuple[str, ...] = ("relu", "tanh", "gelu")
    stability_repeats: int = 3


@dataclass(frozen=True)
class InterpretationConfig:
    """Interpretation settings (occlusion and first-layer maps).

    Attributes:
        occlusion_kernels: Occlusion window sizes.
        occlusion_stride: Sliding stride for the occlusion window.
        occlusion_fill: Fill value on the normalised scale (0.0 = background).
        num_per_class_examples: Representative images per class for occlusion.
    """

    occlusion_kernels: tuple[int, ...] = (1, 2, 3)
    occlusion_stride: int = 1
    occlusion_fill: float = 0.0
    num_per_class_examples: int = 1


@dataclass(frozen=True)
class Config:
    """Top-level experiment configuration.

    Attributes:
        seed: Global random seed for ``random``, ``numpy``, ``torch`` and JAX.
        device: Preferred PyTorch device policy (``auto`` resolves mps/cuda/cpu).
        paths: Filesystem locations.
        quality: Data-quality thresholds.
        split: Train/validation split policy.
        dim_reduction: Dimensionality-reduction settings.
        model: Model architecture.
        training: Optimisation settings.
        grid: Experiment grid.
        interpretation: Interpretation settings.
    """

    seed: int = 42
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    dim_reduction: DimReductionConfig = field(default_factory=DimReductionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    interpretation: InterpretationConfig = field(default_factory=InterpretationConfig)


def _coerce[T](cls: type[T], data: Any) -> T:
    """Recursively build a (possibly nested) dataclass from a mapping.

    Args:
        cls: Target dataclass type.
        data: Mapping of field names to raw values (typically parsed YAML).

    Returns:
        An instance of ``cls`` with nested dataclasses and tuples coerced.

    Raises:
        TypeError: If ``data`` is not a mapping for a dataclass field.
    """
    if not is_dataclass(cls):
        return cast("T", data)
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(data).__name__}")

    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise TypeError(f"Unknown config key '{key}' for {cls.__name__}")
        field_type = known[key].type
        resolved = _resolve_value(field_type, value)
        kwargs[key] = resolved
    return cls(**kwargs)


def _resolve_value(field_type: Any, value: Any) -> Any:
    """Coerce a single field value based on its declared type."""
    type_name = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", "")
    if type_name.endswith("Config") and isinstance(value, dict):
        return _coerce(_CONFIG_TYPES[type_name], value)
    if type_name == "Path":
        return Path(value)
    if isinstance(value, list):
        return tuple(value)
    return value


_CONFIG_TYPES: dict[str, type] = {
    "PathsConfig": PathsConfig,
    "QualityConfig": QualityConfig,
    "SplitConfig": SplitConfig,
    "DimReductionConfig": DimReductionConfig,
    "ModelConfig": ModelConfig,
    "TrainingConfig": TrainingConfig,
    "GridConfig": GridConfig,
    "InterpretationConfig": InterpretationConfig,
}


def load_config(path: str | Path | None = None) -> Config:
    """Load a :class:`Config` from a YAML file, falling back to defaults.

    Args:
        path: Path to a YAML config file. If ``None``, returns default config.

    Returns:
        A fully populated :class:`Config`.
    """
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _coerce(Config, raw)
