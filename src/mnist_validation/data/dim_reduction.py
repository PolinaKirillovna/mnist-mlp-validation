"""Dimensionality reduction for visualisation (PCA, t-SNE, UMAP).

All three projections are computed on the *same* stratified subsample (t-SNE on
60k objects is prohibitively slow), with a fixed seed for reproducibility. The
projected coordinates are returned as plain arrays; marking anomalies/duplicates
and plotting happen in the visualisation layer.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mnist_validation.config import PIXEL_MAX, DimReductionConfig
from mnist_validation.data.loading import DatasetBundle

logger = logging.getLogger(__name__)


def stratified_subsample(bundle: DatasetBundle, size: int, seed: int) -> NDArray[np.int64]:
    """Return sorted positional indices of a stratified subsample.

    Args:
        bundle: Dataset to sample from.
        size: Approximate total subsample size.
        seed: Random seed.

    Returns:
        Sorted array of positional indices into ``bundle``.
    """
    if size >= len(bundle):
        return np.arange(len(bundle), dtype=np.int64)
    rng = np.random.default_rng(seed)
    per_class = size // len(np.unique(bundle.labels))
    chosen: list[NDArray[np.int64]] = []
    for cls in np.unique(bundle.labels):
        cls_idx = np.flatnonzero(bundle.labels == cls)
        take = min(per_class, cls_idx.size)
        chosen.append(rng.choice(cls_idx, size=take, replace=False))
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def _scaled(images: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Scale pixels to ``[0, 1]`` float features."""
    return images.astype(np.float64) / PIXEL_MAX


def pca_projection(images: NDArray[np.uint8], seed: int) -> NDArray[np.float64]:
    """Return a 2-D PCA projection of the images.

    Args:
        images: Image array of shape ``(N, 784)``.
        seed: Random seed for the solver.

    Returns:
        Array of shape ``(N, 2)``.
    """
    coords = PCA(n_components=2, random_state=seed).fit_transform(_scaled(images))
    return cast("NDArray[np.float64]", coords.astype(np.float64))


def tsne_projection(
    images: NDArray[np.uint8], config: DimReductionConfig, seed: int
) -> NDArray[np.float64]:
    """Return a 2-D t-SNE projection (PCA-initialised) of the images.

    Args:
        images: Image array of shape ``(N, 784)``.
        config: Dimensionality-reduction settings.
        seed: Random seed.

    Returns:
        Array of shape ``(N, 2)``.
    """
    pre = PCA(n_components=min(config.pca_components, images.shape[1]), random_state=seed)
    features = pre.fit_transform(_scaled(images))
    tsne = TSNE(
        n_components=2,
        perplexity=config.tsne_perplexity,
        init="pca",
        random_state=seed,
    )
    return cast("NDArray[np.float64]", tsne.fit_transform(features).astype(np.float64))


def umap_projection(
    images: NDArray[np.uint8], config: DimReductionConfig, seed: int
) -> NDArray[np.float64]:
    """Return a 2-D UMAP projection of the images.

    Args:
        images: Image array of shape ``(N, 784)``.
        config: Dimensionality-reduction settings.
        seed: Random seed.

    Returns:
        Array of shape ``(N, 2)``.
    """
    import umap  # local import: pulls in numba, slow to import

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        random_state=seed,
    )
    return cast("NDArray[np.float64]", reducer.fit_transform(_scaled(images)).astype(np.float64))


def compute_projections(
    images: NDArray[np.uint8], config: DimReductionConfig, seed: int
) -> dict[str, NDArray[np.float64]]:
    """Compute PCA, t-SNE and UMAP 2-D projections of the same images.

    Args:
        images: Image array of shape ``(N, 784)``.
        config: Dimensionality-reduction settings.
        seed: Random seed.

    Returns:
        Mapping with keys ``pca``, ``tsne`` and ``umap`` to ``(N, 2)`` arrays.
    """
    logger.info("Computing projections on %d samples", images.shape[0])
    projections = {
        "pca": pca_projection(images, seed),
        "tsne": tsne_projection(images, config, seed),
        "umap": umap_projection(images, config, seed),
    }
    logger.info("Projections computed")
    return projections
