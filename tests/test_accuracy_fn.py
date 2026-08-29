"""Unit tests for ``utils.utils.accuracy_fn``.

Hand-picked y_true / y_pred so the expected percentage is known exactly.
"""

import pytest
import torch

from utils.utils import accuracy_fn


def test_accuracy_fn_all_correct():
    y_true = torch.tensor([0, 1, 2, 3])
    y_pred = torch.tensor([0, 1, 2, 3])
    assert accuracy_fn(y_true, y_pred) == 100.0


def test_accuracy_fn_all_wrong():
    y_true = torch.tensor([0, 1, 2, 3])
    y_pred = torch.tensor([1, 2, 3, 0])
    assert accuracy_fn(y_true, y_pred) == 0.0


def test_accuracy_fn_partial_two_of_three():
    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.tensor([0, 1, 0])  # 2 correct out of 3
    assert accuracy_fn(y_true, y_pred) == pytest.approx(200.0 / 3.0, rel=1e-9)
