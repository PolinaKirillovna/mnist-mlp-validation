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
from numpy.typing import NDArray

from mnist_validation.config import Config
from mnist_validation.data.cleaning import (
    anomalies_frame,
    build_variants,
    save_anomalies,
)
from mnist_validation.data.dim_reduction import compute_projections, stratified_subsample
from mnist_validation.data.loading import DatasetBundle, load_bundle, save_bundle
from mnist_validation.data.quality import QualityReport, collect_findings
from mnist_validation.data.representativeness import (
    qualitative_representativeness,
    quantitative_representativeness,
)
from mnist_validation.data.splitting import image_overlap_row_ids, stratified_train_val_split
from mnist_validation.data.statistics import class_distribution_frame
from mnist_validation.reproducibility import seed_everything
from mnist_validation.visualization import part1 as viz
from mnist_validation.visualization.style import apply_style

logger = logging.getLogger(__name__)

ANOMALIES_FILE = "anomalies.parquet"


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
