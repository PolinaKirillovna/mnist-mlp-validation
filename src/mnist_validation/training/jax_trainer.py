"""JAX/Flax training loop with optax.

Kept deliberately parallel to the PyTorch trainer: identical architecture,
optimiser (AdamW), early-stopping criterion (validation macro-F1) and — crucially
for the comparison — the *same* NumPy-seeded mini-batch order per epoch. The
functional style (explicit params/opt_state, jitted update, PRNG keys for
dropout) is where JAX differs from PyTorch.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray

from mnist_validation.config import NUM_PIXELS, ModelConfig, TrainingConfig
from mnist_validation.data.loading import DatasetBundle
from mnist_validation.evaluation.metrics import macro_f1_from_predictions
from mnist_validation.models.jax_mlp import JaxClassifier, JaxMLP, init_params
from mnist_validation.preprocessing import to_float01
from mnist_validation.reproducibility import batch_permutation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JaxTrainResult:
    """Result of a JAX training run (mirrors the PyTorch ``TrainResult``).

    Attributes:
        classifier: The trained classifier (best checkpoint restored).
        history: Per-epoch metrics.
        best_epoch: Best validation epoch (1-based).
        best_val_macro_f1: Best validation macro-F1.
        num_parameters: Number of trainable parameters.
        train_seconds: Wall-clock training time.
    """

    classifier: JaxClassifier
    history: list[dict[str, float]]
    best_epoch: int
    best_val_macro_f1: float
    num_parameters: int
    train_seconds: float


def _count_params(params: Any) -> int:
    """Return the total number of scalar parameters in a pytree."""
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


@partial(jax.jit, static_argnums=(0, 1))
def _train_step(
    model: JaxMLP,
    optimizer: optax.GradientTransformation,
    params: Any,
    opt_state: Any,
    xb: jax.Array,
    yb: jax.Array,
    dropout_key: jax.Array,
) -> tuple[Any, Any, jax.Array]:
    """Perform a single jitted AdamW update step and return the batch loss."""

    def loss_fn(p: Any) -> jax.Array:
        logits = cast(
            "jax.Array", model.apply({"params": p}, xb, train=True, rngs={"dropout": dropout_key})
        )
        return cast("jax.Array", optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean())

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def _evaluate(
    model: JaxMLP, params: Any, x: jax.Array, y: NDArray[np.int64]
) -> tuple[float, float, float]:
    """Return (mean loss, accuracy, macro-F1) of the model on ``(x, y)``."""
    logits = cast("jax.Array", model.apply({"params": params}, x, train=False))
    loss = float(optax.softmax_cross_entropy_with_integer_labels(logits, jnp.asarray(y)).mean())
    preds = np.asarray(logits.argmax(axis=1), dtype=np.int64)
    accuracy = float((preds == y).mean())
    return loss, accuracy, macro_f1_from_predictions(y, preds)


def train_jax(
    train: DatasetBundle,
    val: DatasetBundle,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    seed: int,
) -> JaxTrainResult:
    """Train a :class:`JaxMLP` and return the best checkpoint with history.

    Args:
        train: Training dataset.
        val: Validation dataset.
        model_config: Model architecture configuration.
        training_config: Optimisation configuration.
        seed: Base seed for parameter init, dropout keys and batch order.

    Returns:
        A :class:`JaxTrainResult`.
    """
    x_train = jnp.asarray(to_float01(train.images))
    x_val = jnp.asarray(to_float01(val.images))
    y_train = train.labels

    model = JaxMLP(
        hidden_sizes=model_config.hidden_sizes,
        activation=model_config.activation,
        dropout_p=model_config.dropout_p,
    )
    params = init_params(model, seed, NUM_PIXELS)
    optimizer = optax.adamw(
        learning_rate=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    opt_state = optimizer.init(params)
    dropout_key = jax.random.PRNGKey(seed + 1)

    history: list[dict[str, float]] = []
    best_params = params
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    n = len(train)
    start_time = time.perf_counter()

    for epoch in range(1, training_config.epochs + 1):
        perm = batch_permutation(n, seed + epoch)
        for start in range(0, n, training_config.batch_size):
            idx = perm[start : start + training_config.batch_size]
            dropout_key, step_key = jax.random.split(dropout_key)
            params, opt_state, _ = _train_step(
                model,
                optimizer,
                params,
                opt_state,
                x_train[idx],
                jnp.asarray(y_train[idx]),
                step_key,
            )

        train_loss, train_acc, train_f1 = _evaluate(model, params, x_train, y_train)
        val_loss, val_acc, val_f1 = _evaluate(model, params, x_val, val.labels)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_params = params
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                logger.info("JAX early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    train_seconds = time.perf_counter() - start_time
    logger.info(
        "Trained JAX %s: best val macro-F1 %.4f at epoch %d (%.1fs)",
        train.name,
        best_val_f1,
        best_epoch,
        train_seconds,
    )
    return JaxTrainResult(
        classifier=JaxClassifier(model, best_params),
        history=history,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_f1,
        num_parameters=_count_params(params),
        train_seconds=train_seconds,
    )
