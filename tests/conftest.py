"""Shared fixtures for the CI test suite.

Everything here is synthetic: random tensors + fixed labels, tiny models.
No real dataset, no filesystem paths into ``data/``, no network, no MLflow
tracking server and no Azure. These tests are meant to run on a plain CPU
GitHub Actions runner.
"""

import random

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.model import tinyConvNeXt

NUM_CLASSES = 3


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic runs so the overfit / shape assertions don't flake."""
    torch.manual_seed(0)
    random.seed(0)


@pytest.fixture
def num_classes():
    return NUM_CLASSES


@pytest.fixture
def dummy_dataloader():
    """12 random 32x32 RGB images with fixed, class-balanced labels.

    - 32x32 (not 128) keeps the CPU forward/backward passes fast; tinyConvNeXt
      pools adaptively so the input size is free.
    - Labels cover every class an equal number of times on purpose:
      ``MulticlassAUROC(average=None)`` returns NaN for any class that never
      appears, and one of the tests asserts "no NaN".
    """
    n_per_class = 4
    images = torch.randn(NUM_CLASSES * n_per_class, 3, 32, 32)
    labels = torch.arange(NUM_CLASSES).repeat(n_per_class)  # [0,1,2,0,1,2,...]
    return DataLoader(TensorDataset(images, labels), batch_size=4)


@pytest.fixture
def tiny_model():
    """Smallest usable tinyConvNeXt: hidden_units=8 runs a full epoch in ms on CPU."""
    return tinyConvNeXt(input_features=3, output_features=NUM_CLASSES, hidden_units=8)
