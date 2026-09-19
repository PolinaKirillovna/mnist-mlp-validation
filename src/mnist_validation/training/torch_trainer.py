"""PyTorch training loop.

A manual loop (rather than a ``DataLoader``) is used so that the mini-batch order
is driven by a shared, seedable NumPy permutation — the same order can then be
reproduced by the JAX trainer for a fair comparison. Early stopping tracks the
validation macro-F1 and restores the best checkpoint.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from mnist_validation.config import ModelConfig, TrainingConfig
from mnist_validation.data.loading import DatasetBundle
from mnist_validation.evaluation.metrics import macro_f1_from_predictions
from mnist_validation.models.torch_mlp import MLP, TorchClassifier, count_parameters
from mnist_validation.preprocessing import to_float01
from mnist_validation.reproducibility import batch_permutation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    """Result of a training run.

    Attributes:
        classifier: The trained classifier (best checkpoint restored).
        history: Per-epoch metrics (loss/accuracy/macro-F1 for train and val).
        best_epoch: Epoch index (1-based) of the best validation macro-F1.
        best_val_macro_f1: Best validation macro-F1 achieved.
        num_parameters: Number of trainable parameters.
        train_seconds: Wall-clock training time in seconds.
    """

    classifier: TorchClassifier
    history: list[dict[str, float]]
    best_epoch: int
    best_val_macro_f1: float
    num_parameters: int
    train_seconds: float


def _evaluate(
    model: MLP,
    x: torch.Tensor,
    y: NDArray[np.int64],
    criterion: nn.Module,
    batch_size: int,
) -> tuple[float, float, float]:
    """Return (mean loss, accuracy, macro-F1) of ``model`` on ``(x, y)``."""
    model.eval()
    losses: list[float] = []
    preds: list[NDArray[np.int64]] = []
    y_tensor = torch.from_numpy(y).to(x.device)
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            yb = y_tensor[start : start + batch_size]
            logits = model(xb)
            losses.append(float(criterion(logits, yb).item()) * xb.shape[0])
            preds.append(logits.argmax(dim=1).cpu().numpy().astype(np.int64))
    y_pred = np.concatenate(preds)
    accuracy = float((y_pred == y).mean())
    macro_f1 = macro_f1_from_predictions(y, y_pred)
    return sum(losses) / x.shape[0], accuracy, macro_f1


def train_torch(
    train: DatasetBundle,
    val: DatasetBundle,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
    seed: int,
) -> TrainResult:
    """Train an :class:`MLP` and return the best checkpoint with history.

    Args:
        train: Training dataset.
        val: Validation dataset.
        model_config: Model architecture configuration.
        training_config: Optimisation configuration.
        device: Device to train on.
        seed: Base seed controlling batch order.

    Returns:
        A :class:`TrainResult`.
    """
    x_train = torch.from_numpy(to_float01(train.images)).to(device)
    x_val = torch.from_numpy(to_float01(val.images)).to(device)
    y_train_np = train.labels
    y_train = torch.from_numpy(y_train_np).to(device)

    model = MLP(
        hidden_sizes=model_config.hidden_sizes,
        activation=model_config.activation,
        dropout_p=model_config.dropout_p,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    n = len(train)
    start_time = time.perf_counter()

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        perm = batch_permutation(n, seed + epoch)
        perm_tensor = torch.from_numpy(perm).to(device)
        for start in range(0, n, training_config.batch_size):
            idx = perm_tensor[start : start + training_config.batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, train_f1 = _evaluate(
            model, x_train, y_train_np, criterion, training_config.batch_size
        )
        val_loss, val_acc, val_f1 = _evaluate(
            model, x_val, val.labels, criterion, training_config.batch_size
        )
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
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    train_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    classifier = TorchClassifier(model, device)
    logger.info(
        "Trained %s: best val macro-F1 %.4f at epoch %d (%.1fs)",
        train.name,
        best_val_f1,
        best_epoch,
        train_seconds,
    )
    return TrainResult(
        classifier=classifier,
        history=history,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_f1,
        num_parameters=count_parameters(model),
        train_seconds=train_seconds,
    )
