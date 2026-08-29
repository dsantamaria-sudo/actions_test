import torch
from torch import nn
from torchvision.models import ConvNeXt_Small_Weights, convnext_small

CONVNEXT_S_WEIGHTS = ConvNeXt_Small_Weights.IMAGENET1K_V1


def create_convnext_s(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = convnext_small(weights=CONVNEXT_S_WEIGHTS)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features=in_features, out_features=num_classes)

    return model


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an NCHW tensor (ConvNeXt normalizes channels-last)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt's building block: 7x7 depthwise conv, then an inverted bottleneck
    (Linear -> GELU -> Linear) applied channels-last, scaled and added back as a residual."""

    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC so LayerNorm/Linear act over channels
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return residual + x


class DownsampleLayer(nn.Module):
    """LayerNorm + stride-2 conv: ConvNeXt's replacement for MaxPool between stages."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class tinyConvNeXt(nn.Module):
    """Same shape as tinyVGG (3 conv blocks + classifier head), but each block
    uses ConvNeXt's depthwise-conv/inverted-bottleneck design instead of plain conv+relu+maxpool.
    Trained from scratch (no pretrained weights) -- input size is flexible since the
    head pools adaptively instead of relying on a hardcoded flattened size."""

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_units: int = 96,
        blocks_per_stage: int = 2,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(input_features, hidden_units, kernel_size=4, stride=4),
            LayerNorm2d(hidden_units),
        )

        self.cnn_block_1 = nn.Sequential(
            *[ConvNeXtBlock(hidden_units) for _ in range(blocks_per_stage)]
        )
        self.downsample_1 = DownsampleLayer(hidden_units, hidden_units * 2)

        self.cnn_block_2 = nn.Sequential(
            *[ConvNeXtBlock(hidden_units * 2) for _ in range(blocks_per_stage)]
        )
        self.downsample_2 = DownsampleLayer(hidden_units * 2, hidden_units * 4)

        self.cnn_block_3 = nn.Sequential(
            *[ConvNeXtBlock(hidden_units * 4) for _ in range(blocks_per_stage)]
        )

        self.class_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(hidden_units * 4),
            nn.Linear(in_features=hidden_units * 4, out_features=output_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.downsample_1(self.cnn_block_1(x))
        x = self.downsample_2(self.cnn_block_2(x))
        x = self.cnn_block_3(x)
        return self.class_layer(x)


class tinyVGG(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=100):

        super().__init__()

        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_features,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.cnn_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.class_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 4 * 4, out_features=output_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.class_layer(self.cnn_block_3(self.cnn_block_2(self.cnn_block_1(x))))
