"""Data layer: loading, quality checks, cleaning and splitting.

This layer knows nothing about the neural-network frameworks. It produces plain
NumPy arrays wrapped in :class:`~mnist_validation.data.loading.DatasetBundle`.
"""

from __future__ import annotations

from mnist_validation.data.loading import DatasetBundle, load_bundle

__all__ = ["DatasetBundle", "load_bundle"]
