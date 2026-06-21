from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

