"""Unit tests for ``model.model``.

- tinyConvNeXt: output shape for batch size 1 and > 1.
- create_convnext_s: backbone freeze / unfreeze behaviour.

``create_convnext_s`` hardcodes ``weights=CONVNEXT_S_WEIGHTS`` (ImageNet), which
would download ~100 MB on first use. The freeze tests monkeypatch that constant
to ``None`` so torchvision builds the same architecture with random init and no
network access -- ``model.features`` and the classifier head are unaffected.
"""

import pytest
import torch

from model.model import create_convnext_s, tinyConvNeXt


@pytest.mark.parametrize("batch_size", [1, 4])
def test_tinyconvnext_output_shape(batch_size):
    model = tinyConvNeXt(input_features=3, output_features=3, hidden_units=96)
    model.eval()

    x = torch.randn(batch_size, 3, 128, 128)  # real input size used in main.py
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, 3)


def test_create_convnext_s_freezes_backbone(monkeypatch):
    monkeypatch.setattr("model.model.CONVNEXT_S_WEIGHTS", None)

    model = create_convnext_s(num_classes=3, freeze_backbone=True)

    assert all(not p.requires_grad for p in model.features.parameters())
    # The freshly swapped classification head must still be trainable.
    assert all(p.requires_grad for p in model.classifier[2].parameters())


def test_create_convnext_s_keeps_backbone_trainable(monkeypatch):
    monkeypatch.setattr("model.model.CONVNEXT_S_WEIGHTS", None)

    model = create_convnext_s(num_classes=3, freeze_backbone=False)

    assert all(p.requires_grad for p in model.features.parameters())
