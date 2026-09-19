"""Model layer: framework-agnostic interface and MLP implementations.

Models only define architecture and forward computation; training lives in the
``training`` layer. All models expose the :class:`~mnist_validation.models.base.Classifier`
protocol so that evaluation and interpretation stay framework-independent.
"""

from __future__ import annotations
