"""Training-grid runner.

Runs the experiment grid (dataset variant x dropout x activation) with identical
optimisation settings, evaluates every run on the shared test set, saves per-run
history/metrics and the best model per dataset, and returns structured records
for the reporting layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from mnist_validation.config import Config, ModelConfig
from mnist_validation.data.loading import DatasetBundle
from mnist_validation.evaluation.metrics import ClassificationMetrics, compute_metrics
from mnist_validation.models.torch_mlp import (
    TorchClassifier,
    resolve_device,
    save_torch_model,
)
from mnist_validation.preprocessing import to_float01
from mnist_validation.reproducibility import seed_everything
from mnist_validation.training.torch_trainer import train_torch

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """One completed grid run.

    Attributes:
        dataset: Dataset-variant name.
        activation: Activation function name.
        dropout: Dropout probability.
        history: Per-epoch training history.
        best_epoch: Best validation epoch (1-based).
        val_macro_f1: Best validation macro-F1.
        num_parameters: Number of trainable parameters.
        train_seconds: Wall-clock training time.
        test_metrics: Metrics on the shared test set.
        classifier: The trained classifier (best checkpoint).
    """

    dataset: str
    activation: str
    dropout: float
    history: list[dict[str, float]]
    best_epoch: int
    val_macro_f1: float
    num_parameters: int
    train_seconds: float
    test_metrics: ClassificationMetrics
    classifier: TorchClassifier

    def model_config(self, hidden_sizes: tuple[int, ...]) -> ModelConfig:
        """Return the :class:`ModelConfig` that produced this run."""
        return ModelConfig(
            hidden_sizes=hidden_sizes, activation=self.activation, dropout_p=self.dropout
        )


@dataclass
class GridResult:
    """Result of the full grid.

    Attributes:
        records: All run records.
        best_by_dataset: Best run per dataset (by validation macro-F1).
    """

    records: list[RunRecord]
    best_by_dataset: dict[str, RunRecord]


def _run_id(dataset: str, activation: str, dropout: float) -> str:
    """Return a filesystem-friendly run identifier."""
    return f"{dataset}_{activation}_p{dropout:g}"


def run_grid(
    variants: dict[str, DatasetBundle],
    val: DatasetBundle,
    test: DatasetBundle,
    config: Config,
) -> GridResult:
    """Run the full training grid and select the best model per dataset.

    Args:
        variants: Mapping of dataset-variant name to training bundle.
        val: Shared validation set.
        test: Shared test set.
        config: Experiment configuration (uses ``grid`` and ``training``).

    Returns:
        A :class:`GridResult`.
    """
    device = resolve_device(config.device)
    logger.info("Training grid on device %s", device)
    x_test = to_float01(test.images).astype("float64")

    runs_dir = config.paths.artifacts_dir / "runs"
    final_dir = config.paths.artifacts_dir / "final"
    records: list[RunRecord] = []

    for dataset in config.grid.datasets:
        train = variants[dataset]
        for activation in config.grid.activations:
            for dropout in config.grid.dropouts:
                seed_everything(config.seed)
                model_config = ModelConfig(
                    hidden_sizes=config.model.hidden_sizes,
                    activation=activation,
                    dropout_p=dropout,
                )
                result = train_torch(train, val, model_config, config.training, device, config.seed)
                test_metrics = compute_metrics(
                    test.labels, result.classifier.predict_logits(x_test)
                )
                record = RunRecord(
                    dataset=dataset,
                    activation=activation,
                    dropout=dropout,
                    history=result.history,
                    best_epoch=result.best_epoch,
                    val_macro_f1=result.best_val_macro_f1,
                    num_parameters=result.num_parameters,
                    train_seconds=result.train_seconds,
                    test_metrics=test_metrics,
                    classifier=result.classifier,
                )
                records.append(record)
                _save_run(record, runs_dir / _run_id(dataset, activation, dropout))

    best_by_dataset = _select_best(records)
    _save_final_models(best_by_dataset, config, final_dir)
    return GridResult(records=records, best_by_dataset=best_by_dataset)


def _select_best(records: list[RunRecord]) -> dict[str, RunRecord]:
    """Return the best run per dataset by validation macro-F1."""
    best: dict[str, RunRecord] = {}
    for rec in records:
        current = best.get(rec.dataset)
        if current is None or rec.val_macro_f1 > current.val_macro_f1:
            best[rec.dataset] = rec
    return best


def _save_run(record: RunRecord, run_dir: Path) -> None:
    """Persist a run's history and scalar metrics as JSON."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(json.dumps(record.history, indent=2), encoding="utf-8")
    metrics: dict[str, object] = dict(record.test_metrics.scalar_summary())
    metrics.update(
        {
            "dataset": record.dataset,
            "activation": record.activation,
            "dropout": record.dropout,
            "val_macro_f1": record.val_macro_f1,
            "best_epoch": float(record.best_epoch),
            "num_parameters": float(record.num_parameters),
            "train_seconds": record.train_seconds,
        }
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _save_final_models(
    best_by_dataset: dict[str, RunRecord], config: Config, final_dir: Path
) -> None:
    """Save the best model per dataset and a selection summary."""
    final_dir.mkdir(parents=True, exist_ok=True)
    selection: dict[str, dict[str, float | str]] = {}
    for dataset, rec in best_by_dataset.items():
        save_torch_model(
            rec.classifier, rec.model_config(config.model.hidden_sizes), final_dir / f"{dataset}.pt"
        )
        selection[dataset] = {
            "activation": rec.activation,
            "dropout": rec.dropout,
            "val_macro_f1": rec.val_macro_f1,
            "test_accuracy": rec.test_metrics.accuracy,
            "test_macro_f1": rec.test_metrics.macro_f1,
        }
    (final_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d final models to %s", len(best_by_dataset), final_dir)
