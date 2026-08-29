"""Unit tests for ``utils.utils.train_step`` / ``utils.utils.test_step``.

These call the two step functions directly in a plain Python loop -- ``train()``
is deliberately not touched because it opens an MLflow connection.
"""

import math

import torch
from torch import nn

# Aliased so pytest does not collect `test_step` itself as a test case.
from utils.utils import test_step as run_test_step
from utils.utils import train_step as run_train_step

# The scalar metrics every step dict must expose (all Python floats).
SCALAR_KEYS = ["loss", "acc", "precision", "recall", "f1", "auroc"]
# The per-class variants (lists of length num_classes).
PER_CLASS_KEYS = ["precision_per_class", "recall_per_class", "f1_per_class", "auroc_per_class"]
EXPECTED_KEYS = set(SCALAR_KEYS) | set(PER_CLASS_KEYS)


def _assert_valid_metrics(result, num_classes):
    """Same shape/validity checks for both train_step and test_step output."""
    assert set(result) == EXPECTED_KEYS

    for key in SCALAR_KEYS:
        value = result[key]
        assert isinstance(value, float)
        assert math.isfinite(value), f"{key} is not finite: {value}"

    for key in PER_CLASS_KEYS:
        values = result[key]
        assert isinstance(values, list)
        assert len(values) == num_classes
        for i, value in enumerate(values):
            assert isinstance(value, float)
            assert math.isfinite(value), f"{key}[{i}] is not finite: {value}"

    # Sanity ranges: metrics are ratios, acc is a percentage, loss is non-negative.
    for key in ["precision", "recall", "f1", "auroc"]:
        assert 0.0 <= result[key] <= 1.0
    assert 0.0 <= result["acc"] <= 100.0
    assert result["loss"] >= 0.0


def test_train_step_overfits_small_batch(tiny_model, dummy_dataloader, num_classes):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-2)

    losses = []
    last_result = None
    for _ in range(20):
        last_result = run_train_step(
            model=tiny_model,
            dataloader=dummy_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            num_classes=num_classes,
            device="cpu",
        )
        losses.append(last_result["loss"])

    # Random images + fixed labels are memorizable, so training loss must drop.
    assert losses[-1] < losses[0]
    _assert_valid_metrics(last_result, num_classes)


def test_test_step_runs_without_grad(tiny_model, dummy_dataloader, num_classes):
    loss_fn = nn.CrossEntropyLoss()

    # Start from a clean grad state so we can prove test_step never populated it.
    for param in tiny_model.parameters():
        param.grad = None

    result = run_test_step(
        model=tiny_model,
        dataloader=dummy_dataloader,
        loss_fn=loss_fn,
        num_classes=num_classes,
        device="cpu",
    )

    _assert_valid_metrics(result, num_classes)

    # Default call must not add the confusion matrix entry.
    assert "confusion_matrix" not in result

    # test_step calls model.eval() and runs under torch.inference_mode():
    # the module is left in eval mode and no gradients were computed.
    assert tiny_model.training is False
    assert all(param.grad is None for param in tiny_model.parameters())
