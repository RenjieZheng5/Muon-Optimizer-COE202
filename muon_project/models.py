from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        hidden_dim: int = 512,
        depth: int = 2,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = input_dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=use_bias))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SmallCNN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        width: int = 32,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        channels = input_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=use_bias),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1, bias=use_bias),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(width * 2 * 4 * 4, num_classes, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(
    name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    use_bias: bool = False,
) -> nn.Module:
    if name == "mlp":
        return MLP(input_shape=input_shape, num_classes=num_classes, use_bias=use_bias)
    if name == "small_cnn":
        return SmallCNN(input_shape=input_shape, num_classes=num_classes, use_bias=use_bias)
    raise ValueError(f"Unknown model: {name}")
