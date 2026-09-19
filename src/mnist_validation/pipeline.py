"""High-level pipeline orchestration.

Functions here wire the data/model/evaluation layers into the end-to-end stages
invoked by the CLI. They own side effects (writing tables, figures, processed
datasets); the underlying layers stay pure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from mnist_validation.config import Config, ModelConfig
from mnist_validation.data.cleaning import (
    anomalies_frame,
    build_variants,
    load_anomalies,
    save_anomalies,
)
from mnist_validation.data.dim_reduction import compute_projections, stratified_subsample
from mnist_validation.data.loading import (
    DatasetBundle,
    load_bundle,
    load_bundle_npz,
    save_bundle,
)
from mnist_validation.data.quality import QualityReport, collect_findings
from mnist_validation.data.representativeness import (
    qualitative_representativeness,
    quantitative_representativeness,
)
from mnist_validation.data.splitting import image_overlap_row_ids, stratified_train_val_split
from mnist_validation.data.statistics import class_distribution_frame
from mnist_validation.evaluation.error_analysis import (
    analyze,
    anomaly_behavior_table,
    model_disagreement_indices,
    select_example_indices,
    top_confused_pairs,
)
from mnist_validation.evaluation.metrics import compute_metrics
from mnist_validation.interpretation import gradients as grads
from mnist_validation.interpretation.first_layer import mean_activation_by_class, weight_maps
from mnist_validation.interpretation.occlusion import occlusion_map
from mnist_validation.models.base import Classifier
from mnist_validation.models.torch_mlp import load_torch_model, resolve_device
from mnist_validation.preprocessing import to_float01
from mnist_validation.reproducibility import seed_everything
from mnist_validation.training.grid_runner import GridResult, run_grid
from mnist_validation.training.jax_trainer import train_jax
from mnist_validation.training.torch_trainer import train_torch
from mnist_validation.visualization import part1 as viz
from mnist_validation.visualization import part2 as viz2
from mnist_validation.visualization import part3 as viz3
from mnist_validation.visualization import part4 as viz4
from mnist_validation.visualization.style import apply_style

logger = logging.getLogger(__name__)

ANOMALIES_FILE = "anomalies.parquet"
PROCESSED_VARIANTS = ("original", "balanced", "cleaned")


def _write_table(frame: pd.DataFrame, tables_dir: Path, name: str) -> None:
    """Write a dataframe to ``tables_dir/name`` as CSV."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables_dir / name, index=False)
    logger.info("Wrote table %s (%d rows)", name, len(frame))


def _anomalies_by_reason(report: QualityReport, total: int) -> pd.DataFrame:
    """Build a per-reason count table plus the overall anomaly fraction."""
    summary = report.summary_frame()
    anomaly_total = int(report.anomaly_row_ids().size)
    summary["fraction_of_dataset"] = summary["count"] / total
    overall = pd.DataFrame(
        {
            "reason": ["ANY_ANOMALY"],
            "description": ["Any anomaly reason (union, unique rows)"],
            "count": [anomaly_total],
            "fraction_of_dataset": [anomaly_total / total],
        }
    )
    return pd.concat([summary, overall], ignore_index=True)


def _stats_by_class(report: QualityReport) -> pd.DataFrame:
    """Mean per-class brightness/fill statistics."""
    cols = ["mean_intensity", "std_intensity", "active_fraction", "intensity_sum", "nonzero_count"]
    return report.stats.groupby("label")[cols].mean().reset_index()


def _make_masks(
    subsample_ids: NDArray[np.int64], report: QualityReport
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Boolean anomaly/duplicate masks for a subsample (row_id == position)."""
    anomaly_set = set(report.anomaly_row_ids().tolist())
    dup_set = set(report.duplicate_row_ids().tolist())
    anomaly_mask = np.array([int(i) in anomaly_set for i in subsample_ids], dtype=bool)
    dup_mask = np.array([int(i) in dup_set for i in subsample_ids], dtype=bool)
    return anomaly_mask, dup_mask


def run_validation(config: Config) -> dict[str, object]:
    """Run the full Part 1 data-validation pipeline.

    Loads the training variant and shared test set, runs all quality checks,
    computes representativeness, builds the three training variants, saves
    anomalies and processed datasets, and writes all Part 1 tables and figures.

    Args:
        config: Loaded experiment configuration.

    Returns:
        A summary dictionary (also written to ``reports/tables/part1_summary.json``).
    """
    seed_everything(config.seed)
    apply_style()
    paths = config.paths
    tables = paths.tables_dir
    figures = paths.figures_dir

    train_full = load_bundle(paths.train_path, "train")
    test = load_bundle(paths.test_path, "test")

    report = collect_findings(
        train_full, config.quality, config.seed, config.dim_reduction.pca_components
    )

    # --- Tables ---------------------------------------------------------------
    dist = class_distribution_frame(train_full.labels)
    _write_table(dist, tables, "part1_class_distribution.csv")
    _write_table(report.summary_frame(), tables, "part1_quality_summary.csv")
    _write_table(
        _anomalies_by_reason(report, len(train_full)), tables, "part1_anomalies_by_reason.csv"
    )
    quant = quantitative_representativeness(train_full.labels)
    _write_table(quant, tables, "part1_representativeness.csv")
    _write_table(_stats_by_class(report), tables, "part1_stats_by_class.csv")

    qual = qualitative_representativeness(train_full.labels)

    # --- Split, variants, processed datasets ----------------------------------
    train_raw, val = stratified_train_val_split(train_full, config.split.val_fraction, config.seed)
    variants = build_variants(train_raw, report, config.seed)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, bundle in variants.items():
        save_bundle(bundle, paths.processed_dir / f"{name}.npz")
    save_bundle(val, paths.processed_dir / "val.npz")
    save_bundle(test, paths.processed_dir / "test.npz")

    sizes = {name: len(bundle) for name, bundle in variants.items()}
    _write_table(
        pd.DataFrame({"dataset": list(sizes), "size": list(sizes.values())}),
        tables,
        "part1_dataset_sizes.csv",
    )

    # --- Anomalies persistence ------------------------------------------------
    anomalies = anomalies_frame(train_full, report)
    save_anomalies(anomalies, paths.anomalies_dir / ANOMALIES_FILE)

    # --- Train-test leakage ---------------------------------------------------
    overlap = image_overlap_row_ids(train_full, test)
    _write_table(
        pd.DataFrame({"test_row_id_in_train": overlap}),
        tables,
        "part1_test_train_overlap.csv",
    )

    # --- Figures --------------------------------------------------------------
    viz.plot_class_examples(train_full.images, train_full.labels, figures / "part1_examples.png")
    viz.plot_class_histogram(train_full.labels, figures / "part1_class_histogram.png")
    viz.plot_balance_comparison(
        train_raw.labels, variants["balanced"].labels, figures / "part1_balance_compare.png"
    )
    viz.plot_dataset_sizes(sizes, figures / "part1_dataset_sizes.png")
    viz.plot_intensity_boxplot(report.stats, "mean_intensity", figures / "part1_intensity_mean.png")
    viz.plot_intensity_boxplot(report.stats, "active_fraction", figures / "part1_fill.png")

    sub = stratified_subsample(train_full, config.dim_reduction.subsample_size, config.seed)
    projections = compute_projections(train_full.images[sub], config.dim_reduction, config.seed)
    anomaly_mask, dup_mask = _make_masks(train_full.row_ids[sub], report)
    titles = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
    for key, coords in projections.items():
        viz.plot_projection(
            coords,
            train_full.labels[sub],
            anomaly_mask,
            dup_mask,
            f"Проекция {titles[key]}",
            figures / f"part1_{key}.png",
        )

    _plot_example_galleries(train_full, report, figures)

    # --- Summary --------------------------------------------------------------
    summary: dict[str, object] = {
        "n_train_full": len(train_full),
        "n_test": len(test),
        "class_counts": np.bincount(train_full.labels, minlength=10).tolist(),
        "n_full_duplicates": report.result("full_duplicate").count,
        "n_label_conflicts": report.result("label_conflict").count,
        "n_anomalies": int(report.anomaly_row_ids().size),
        "anomaly_fraction": float(report.anomaly_row_ids().size / len(train_full)),
        "imbalance_ratio": qual.imbalance_ratio,
        "entropy_normalized": qual.entropy_normalized,
        "min_class": qual.min_class,
        "max_class": qual.max_class,
        "variant_sizes": sizes,
        "val_size": len(val),
        "test_train_overlap": int(overlap.size),
        "representativeness_sufficient": quant["sufficient"].all().item(),
    }
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "part1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Part 1 summary: %s", summary)
    return summary


def _plot_example_galleries(
    train_full: DatasetBundle, report: QualityReport, figures: Path
) -> None:
    """Plot example galleries for duplicates and anomalies with reason captions."""
    dup_ids = report.duplicate_row_ids()
    if dup_ids.size:
        picks = dup_ids[: min(12, dup_ids.size)]
        images = train_full.images[picks]
        captions = [f"класс {int(train_full.labels[p])}" for p in picks]
        viz.plot_examples_with_captions(
            images, captions, "Примеры дубликатов", figures / "part1_duplicates.png"
        )

    anomaly_ids = report.anomaly_row_ids()
    if anomaly_ids.size:
        picks = anomaly_ids[: min(12, anomaly_ids.size)]
        images = train_full.images[picks]
        captions = [
            viz.reasons_to_caption(";".join(sorted(report.reasons_by_row[int(p)]))) for p in picks
        ]
        viz.plot_examples_with_captions(
            images, captions, "Примеры аномальных объектов", figures / "part1_anomalies.png"
        )


def load_processed_variants(
    config: Config,
) -> tuple[dict[str, DatasetBundle], DatasetBundle, DatasetBundle]:
    """Load processed training variants plus the shared val/test bundles.

    Runs Part 1 first if the processed datasets are missing.

    Args:
        config: Loaded experiment configuration.

    Returns:
        A ``(variants, val, test)`` tuple.
    """
    processed = config.paths.processed_dir
    needed = [processed / f"{name}.npz" for name in (*PROCESSED_VARIANTS, "val", "test")]
    if not all(path.exists() for path in needed):
        logger.info("Processed datasets missing; running Part 1 validation first")
        run_validation(config)
    variants = {name: load_bundle_npz(processed / f"{name}.npz") for name in PROCESSED_VARIANTS}
    val = load_bundle_npz(processed / "val.npz")
    test = load_bundle_npz(processed / "test.npz")
    return variants, val, test


def _grid_summary_frame(result: GridResult) -> pd.DataFrame:
    """Build the full grid summary dataframe (one row per run)."""
    rows = []
    for rec in result.records:
        m = rec.test_metrics
        rows.append(
            {
                "dataset": rec.dataset,
                "activation": rec.activation,
                "dropout": rec.dropout,
                "val_macro_f1": rec.val_macro_f1,
                "test_accuracy": m.accuracy,
                "test_macro_f1": m.macro_f1,
                "test_weighted_f1": m.weighted_f1,
                "test_error_rate": m.error_rate,
                "mean_conf_correct": m.mean_confidence_correct,
                "mean_conf_incorrect": m.mean_confidence_incorrect,
                "best_epoch": rec.best_epoch,
                "num_parameters": rec.num_parameters,
                "train_seconds": round(rec.train_seconds, 2),
            }
        )
    return pd.DataFrame(rows)


def _final_metrics_frames(result: GridResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (scalar per-dataset, per-class F1) tables for the three final models."""
    scalar_rows = []
    per_class_rows = []
    for dataset, rec in result.best_by_dataset.items():
        m = rec.test_metrics
        scalar_rows.append(
            {
                "dataset": dataset,
                "activation": rec.activation,
                "dropout": rec.dropout,
                **m.scalar_summary(),
            }
        )
        for cls, f1 in enumerate(rec.test_metrics.f1_per_class.tolist()):
            per_class_rows.append({"dataset": dataset, "class": cls, "f1": f1})
    return pd.DataFrame(scalar_rows), pd.DataFrame(per_class_rows)


def run_grid_pipeline(config: Config) -> dict[str, object]:
    """Run Part 2: the full training grid, then write tables and figures.

    Args:
        config: Loaded experiment configuration.

    Returns:
        A summary dictionary (also written to ``reports/tables/part2_summary.json``).
    """
    seed_everything(config.seed)
    apply_style()
    tables = config.paths.tables_dir
    figures = config.paths.figures_dir

    variants, val, test = load_processed_variants(config)
    result = run_grid(variants, val, test, config)

    grid_frame = _grid_summary_frame(result)
    _write_table(grid_frame, tables, "part2_grid.csv")
    scalar_frame, per_class_frame = _final_metrics_frames(result)
    _write_table(scalar_frame, tables, "part2_final_metrics.csv")
    _write_table(per_class_frame, tables, "part2_final_per_class_f1.csv")

    records = [
        {
            "dataset": r.dataset,
            "activation": r.activation,
            "dropout": r.dropout,
            "history": r.history,
        }
        for r in result.records
    ]
    viz2.plot_grid_learning_curves(
        records, config.grid.datasets, config.grid.activations, figures / "part2_grid_curves.png"
    )
    viz2.plot_hyperparameter_effect(grid_frame, figures / "part2_hyperparams.png")
    viz2.plot_metric_bars(_test_metric_bars(result), figures / "part2_metric_bars.png")
    for dataset, rec in result.best_by_dataset.items():
        viz2.plot_final_learning_curves(
            rec.history,
            f"Кривые обучения — {viz2.DATASET_RU.get(dataset, dataset)} "
            f"(p={rec.dropout:g}, {rec.activation})",
            figures / f"part2_curves_{dataset}.png",
        )
        viz2.plot_confusion_matrix(
            rec.test_metrics.confusion,
            f"Матрица ошибок — {viz2.DATASET_RU.get(dataset, dataset)}",
            figures / f"part2_confusion_{dataset}.png",
        )

    summary = _grid_summary_dict(result)
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "part2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Part 2 summary: %s", summary)
    return summary


def _test_metric_bars(result: GridResult) -> pd.DataFrame:
    """Return per-dataset test metrics for the best runs (for the bar chart)."""
    rows = []
    for dataset, rec in result.best_by_dataset.items():
        m = rec.test_metrics
        rows.append(
            {
                "dataset": dataset,
                "test_accuracy": m.accuracy,
                "test_macro_f1": m.macro_f1,
                "test_weighted_f1": m.weighted_f1,
            }
        )
    return pd.DataFrame(rows)


def _grid_summary_dict(result: GridResult) -> dict[str, object]:
    """Build a compact JSON-serialisable summary of the grid."""
    best = {
        dataset: {
            "activation": rec.activation,
            "dropout": rec.dropout,
            "val_macro_f1": round(rec.val_macro_f1, 4),
            "test_accuracy": round(rec.test_metrics.accuracy, 4),
            "test_macro_f1": round(rec.test_metrics.macro_f1, 4),
        }
        for dataset, rec in result.best_by_dataset.items()
    }
    return {
        "n_runs": len(result.records),
        "total_train_seconds": round(sum(r.train_seconds for r in result.records), 1),
        "num_parameters": result.records[0].num_parameters if result.records else 0,
        "best_by_dataset": best,
    }


# --- Part 3: interpretation --------------------------------------------------

QUADRANT_RU = {
    "correct_high": "верно, высокая уверенность",
    "correct_low": "верно, низкая уверенность",
    "incorrect_high": "ошибка, высокая уверенность",
    "incorrect_low": "ошибка, низкая уверенность",
}


def load_final_models(
    config: Config,
) -> tuple[dict[str, Classifier], dict[str, str], torch.device]:
    """Load the three final models and their activations from ``artifacts/final``.

    Args:
        config: Loaded experiment configuration.

    Returns:
        A ``(classifiers, activations, device)`` tuple; runs the grid first if the
        final models are missing.
    """
    device = resolve_device(config.device)
    final_dir = config.paths.artifacts_dir / "final"
    if not (final_dir / "selection.json").exists():
        logger.info("Final models missing; running Part 2 grid first")
        run_grid_pipeline(config)
    selection = json.loads((final_dir / "selection.json").read_text(encoding="utf-8"))
    classifiers: dict[str, Classifier] = {}
    activations: dict[str, str] = {}
    for dataset in PROCESSED_VARIANTS:
        classifiers[dataset] = load_torch_model(final_dir / f"{dataset}.pt", device)
        activations[dataset] = selection[dataset]["activation"]
    return classifiers, activations, device


def _example_caption(true_label: int, pred: int, conf: float) -> str:
    """Return a short Russian caption for an example image."""
    return f"ист.{true_label}/пред.{pred}\np={conf:.2f}"


def _occlusion_targets(
    test: DatasetBundle, primary_report: object, anomalies: DatasetBundle
) -> list[tuple[str, int, NDArray[np.uint8], int]]:
    """Select representative images for occlusion (per-class, errors, anomalies)."""
    targets: list[tuple[str, int, NDArray[np.uint8], int]] = []
    for cls in range(10):
        pos = int(np.flatnonzero(test.labels == cls)[0])
        targets.append((f"class{cls}", pos, test.images[pos], int(test.labels[pos])))
    incorrect = select_example_indices(primary_report, n=2)["incorrect_high"]  # type: ignore[arg-type]
    for i, pos in enumerate(incorrect.tolist()):
        targets.append((f"error{i}", int(pos), test.images[int(pos)], int(test.labels[int(pos)])))
    for i in range(min(2, len(anomalies))):
        targets.append((f"anomaly{i}", -1, anomalies.images[i], int(anomalies.labels[i])))
    return targets


def run_interpretation(config: Config) -> dict[str, object]:
    """Run Part 3: error analysis, occlusion and first-layer interpretation.

    Args:
        config: Loaded experiment configuration.

    Returns:
        A summary dictionary (also written to ``reports/tables/part3_summary.json``).
    """
    seed_everything(config.seed)
    apply_style()
    tables = config.paths.tables_dir
    figures = config.paths.figures_dir

    classifiers, activations, _ = load_final_models(config)
    test = load_bundle_npz(config.paths.processed_dir / "test.npz")
    anomaly_bundle, reasons = load_anomalies(config.paths.anomalies_dir / ANOMALIES_FILE)

    reports = {name: analyze(clf, test) for name, clf in classifiers.items()}

    # Example galleries + confused pairs per model.
    confused_rows = []
    for name, report in reports.items():
        quadrants = select_example_indices(report, n=8)
        for quad, idx in quadrants.items():
            if idx.size == 0:
                continue
            captions = [
                _example_caption(
                    int(report.labels[p]), int(report.predictions[p]), float(report.confidence[p])
                )
                for p in idx
            ]
            viz3.plot_example_gallery(
                test.images[idx],
                captions,
                f"{viz3.DATASET_RU.get(name, name)}: {QUADRANT_RU[quad]}",
                figures / f"part3_examples_{name}_{quad}.png",
            )
        metrics = compute_metrics(
            test.labels,
            classifiers[name].predict_logits(to_float01(test.images).astype(np.float64)),
        )
        for true_c, pred_c, count in top_confused_pairs(metrics.confusion, top_n=5):
            confused_rows.append(
                {"dataset": name, "true_class": true_c, "predicted_class": pred_c, "count": count}
            )
    _write_table(pd.DataFrame(confused_rows), tables, "part3_confused_pairs.csv")

    # Cross-model disagreement gallery.
    disagree = model_disagreement_indices(list(reports.values()))
    if disagree.size:
        picks = disagree[:16]
        captions = [
            "/".join(str(int(reports[name].predictions[p])) for name in classifiers)
            + f"\nист.{int(test.labels[p])}"
            for p in picks
        ]
        viz3.plot_example_gallery(
            test.images[picks],
            captions,
            "Объекты, на которых модели расходятся (пред. исх/сбаланс/очищ)",
            figures / "part3_disagreement.png",
        )

    # Behaviour on anomalies.
    behavior = anomaly_behavior_table(classifiers, anomaly_bundle, reasons)
    _write_table(behavior, tables, "part3_anomaly_behavior.csv")

    # Occlusion maps for representative images.
    primary = reports["original"]
    kernels = config.interpretation.occlusion_kernels
    for tag, _pos, image, true_label in _occlusion_targets(test, primary, anomaly_bundle):
        norm = to_float01(image.reshape(1, -1)).astype(np.float64)[0]
        maps: dict[str, dict[int, NDArray[np.float64]]] = {}
        for name, clf in classifiers.items():
            maps[name] = {
                k: occlusion_map(
                    clf,
                    norm,
                    true_label,
                    k,
                    config.interpretation.occlusion_stride,
                    config.interpretation.occlusion_fill,
                ).importance_pred
                for k in kernels
            }
        viz3.plot_occlusion_grid(
            norm,
            maps,
            kernels,
            f"Окклюзия — {tag} (истинный класс {true_label})",
            figures / f"part3_occlusion_{tag}.png",
        )

    # First-layer weight maps and activation heatmaps.
    x_test = to_float01(test.images).astype(np.float64)
    for name, clf in classifiers.items():
        viz3.plot_weight_maps(
            weight_maps(clf),
            f"Карты весов первого слоя — {viz3.DATASET_RU.get(name, name)}",
            figures / f"part3_weights_{name}.png",
        )
        heatmap = mean_activation_by_class(clf, x_test, test.labels, activations[name])
        viz3.plot_activation_heatmap(
            heatmap,
            f"Средняя активация нейронов по классам — {viz3.DATASET_RU.get(name, name)}",
            figures / f"part3_activation_{name}.png",
        )

    summary: dict[str, object] = {
        "n_disagreement": int(disagree.size),
        "disagreement_fraction": float(disagree.size / len(test)),
        "n_anomalies_evaluated": len(anomaly_bundle),
        "anomaly_mean_agreement": float(behavior["models_matching_original"].mean()),
        "confused_pairs": confused_rows[:5],
    }
    (tables / "part3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Part 3 summary: %s", summary)
    return summary


# --- Part 4: PyTorch vs JAX comparison + gradient methods ---------------------

GRADIENT_METHODS = ("saliency", "input_x_gradient", "integrated_gradients")


def _pearson(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Return the Pearson correlation between two flattened maps."""
    a = a.ravel()
    b = b.ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _final_configs(config: Config) -> dict[str, ModelConfig]:
    """Return the winning (activation, dropout) config per dataset from the grid."""
    final_dir = config.paths.artifacts_dir / "final"
    if not (final_dir / "selection.json").exists():
        run_grid_pipeline(config)
    selection = json.loads((final_dir / "selection.json").read_text(encoding="utf-8"))
    return {
        dataset: ModelConfig(
            hidden_sizes=config.model.hidden_sizes,
            activation=selection[dataset]["activation"],
            dropout_p=float(selection[dataset]["dropout"]),
        )
        for dataset in PROCESSED_VARIANTS
    }


def run_jax_comparison(config: Config) -> dict[str, object]:
    """Run Part 4: JAX runs, framework comparison, stability and gradient methods.

    Args:
        config: Loaded experiment configuration.

    Returns:
        A summary dictionary (also written to ``reports/tables/part4_summary.json``).
    """
    seed_everything(config.seed)
    apply_style()
    tables = config.paths.tables_dir
    figures = config.paths.figures_dir

    variants, val, test = load_processed_variants(config)
    device = resolve_device(config.device)
    final_configs = _final_configs(config)
    x_test = to_float01(test.images).astype(np.float64)
    seeds = [config.seed + i for i in range(config.grid.stability_repeats)]

    stability_rows: list[dict[str, object]] = []
    base_torch: dict[str, object] = {}
    base_jax: dict[str, object] = {}

    for dataset in PROCESSED_VARIANTS:
        mc = final_configs[dataset]
        for i, s in enumerate(seeds):
            seed_everything(s)
            tr = train_torch(variants[dataset], val, mc, config.training, device, s)
            tm = compute_metrics(test.labels, tr.classifier.predict_logits(x_test))
            seed_everything(s)
            jr = train_jax(variants[dataset], val, mc, config.training, s)
            jm = compute_metrics(test.labels, jr.classifier.predict_logits(x_test))
            for fw, res, met in (("PyTorch", tr, tm), ("JAX", jr, jm)):
                stability_rows.append(
                    {
                        "dataset": dataset,
                        "framework": fw,
                        "seed": s,
                        "test_macro_f1": met.macro_f1,
                        "test_accuracy": met.accuracy,
                        "train_seconds": res.train_seconds,
                    }
                )
            if i == 0:
                base_torch[dataset] = (tr, tm)
                base_jax[dataset] = (jr, jm)

    stability = pd.DataFrame(stability_rows)
    _write_table(stability, tables, "part4_stability_runs.csv")

    # Comparison table (base-seed run) + stability mean/std.
    comparison_rows: list[dict[str, object]] = []
    for dataset in PROCESSED_VARIANTS:
        for fw, store in (("PyTorch", base_torch), ("JAX", base_jax)):
            res, met = store[dataset]  # type: ignore[misc]
            sub = stability[(stability["dataset"] == dataset) & (stability["framework"] == fw)]
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "framework": fw,
                    "activation": final_configs[dataset].activation,
                    "dropout": final_configs[dataset].dropout_p,
                    "test_accuracy": met.accuracy,
                    "test_macro_f1": met.macro_f1,
                    "macro_f1_mean": float(sub["test_macro_f1"].mean()),
                    "macro_f1_std": float(sub["test_macro_f1"].std(ddof=0)),
                    "train_seconds_mean": float(sub["train_seconds"].mean()),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    _write_table(comparison, tables, "part4_frameworks.csv")

    # Characteristic-error overlap (base seed) between frameworks.
    error_rows = []
    for dataset in PROCESSED_VARIANTS:
        t_pred = base_torch[dataset][0].classifier.predict_logits(x_test).argmax(1)  # type: ignore[index]
        j_pred = base_jax[dataset][0].classifier.predict_logits(x_test).argmax(1)  # type: ignore[index]
        t_err = set(np.flatnonzero(t_pred != test.labels).tolist())
        j_err = set(np.flatnonzero(j_pred != test.labels).tolist())
        union = t_err | j_err
        inter = t_err & j_err
        error_rows.append(
            {
                "dataset": dataset,
                "torch_errors": len(t_err),
                "jax_errors": len(j_err),
                "shared_errors": len(inter),
                "error_jaccard": len(inter) / len(union) if union else 1.0,
                "prediction_agreement": float((t_pred == j_pred).mean()),
            }
        )
    _write_table(pd.DataFrame(error_rows), tables, "part4_error_overlap.csv")

    stab_summary = (
        stability.groupby(["dataset", "framework"])["test_macro_f1"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Figures: metric bars, stability, learning-curve overlays.
    viz4.plot_framework_metric_bars(comparison, figures / "part4_framework_bars.png")
    viz4.plot_stability(stab_summary, figures / "part4_stability.png")
    for dataset in PROCESSED_VARIANTS:
        viz4.plot_lr_overlay(
            base_torch[dataset][0].history,  # type: ignore[index]
            base_jax[dataset][0].history,  # type: ignore[index]
            f"PyTorch vs JAX — {viz4.DATASET_RU.get(dataset, dataset)}",
            figures / f"part4_curves_{dataset}.png",
        )

    # JAX first-layer weight maps (compare with the PyTorch maps from Part 3).
    for dataset in PROCESSED_VARIANTS:
        viz4_clf = base_jax[dataset][0].classifier  # type: ignore[index]
        viz3.plot_weight_maps(
            weight_maps(viz4_clf),
            f"Карты весов первого слоя (JAX) — {viz4.DATASET_RU.get(dataset, dataset)}",
            figures / f"part4_weights_jax_{dataset}.png",
        )

    # Gradient methods + map-correlation analysis on representative images.
    corr_summary = _gradient_analysis(config, base_torch, base_jax, test, figures, tables)

    summary: dict[str, object] = {
        "stability_repeats": config.grid.stability_repeats,
        "comparison": comparison_rows,
        "error_overlap": error_rows,
        "map_correlation": corr_summary,
    }
    (tables / "part4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Part 4 summary written")
    return summary


def _gradient_analysis(
    config: Config,
    base_torch: dict[str, object],
    base_jax: dict[str, object],
    test: DatasetBundle,
    figures: Path,
    tables: Path,
) -> dict[str, float]:
    """Compute gradient attributions, compare frameworks and compare to occlusion."""
    sample_positions = [int(np.flatnonzero(test.labels == c)[0]) for c in (0, 3, 8, 9)]
    dataset = "balanced"
    torch_clf = base_torch[dataset][0].classifier  # type: ignore[index]
    jax_clf = base_jax[dataset][0].classifier  # type: ignore[index]
    torch_grad = grads.torch_input_gradient(torch_clf)
    jax_grad = grads.jax_input_gradient(jax_clf)

    corr_saliency: list[float] = []
    corr_occ_vs_grad: list[float] = []
    rows: list[dict[str, object]] = []
    for pos in sample_positions:
        image = to_float01(test.images[pos : pos + 1]).astype(np.float64)[0]
        target = int(test.labels[pos])
        t_maps = grads.all_methods(torch_grad, image, target)
        j_maps = grads.all_methods(jax_grad, image, target)
        viz4.plot_gradient_methods(
            image,
            {"PyTorch": t_maps, "JAX": j_maps},
            GRADIENT_METHODS,
            f"Градиентные карты — класс {target}",
            figures / f"part4_gradients_class{target}.png",
        )
        occ = occlusion_map(torch_clf, image, target, kernel=3).importance_true
        c_sal = _pearson(t_maps["saliency"], j_maps["saliency"])
        c_og = _pearson(occ, t_maps["saliency"].reshape(occ.shape))
        corr_saliency.append(c_sal)
        corr_occ_vs_grad.append(c_og)
        rows.append(
            {
                "class": target,
                "corr_saliency_torch_jax": c_sal,
                "corr_occlusion_vs_saliency": c_og,
            }
        )
    _write_table(pd.DataFrame(rows), tables, "part4_map_correlation.csv")
    return {
        "mean_corr_saliency_torch_jax": float(np.mean(corr_saliency)),
        "mean_corr_occlusion_vs_saliency": float(np.mean(corr_occ_vs_grad)),
    }
