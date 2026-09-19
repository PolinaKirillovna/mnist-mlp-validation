"""Reproducibility helpers.

A single entry point that seeds every source of randomness used in the project
(``random``, ``numpy``, ``torch`` when available) and returns a base JAX-style
key seed. Framework-specific seeding lives close to the frameworks; this module
only touches what is always importable.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if importable) PyTorch for deterministic runs.

    Args:
        seed: The global seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch always present in this project
        logger.debug("torch not importable while seeding; skipping torch seed")


def batch_permutation(num_samples: int, seed: int) -> NDArray[np.int64]:
    """Return a deterministic permutation of sample indices.

    Shared by both the PyTorch and JAX trainers so that the mini-batch order is
    identical across frameworks for a given epoch seed.

    Args:
        num_samples: Number of samples to permute.
        seed: Seed for this particular permutation (e.g. base seed + epoch).

    Returns:
        A permuted array of indices ``[0, num_samples)``.
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(num_samples)
