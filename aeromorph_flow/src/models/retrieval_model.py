from __future__ import annotations

import torch
from torch import nn

from aeromorph_flow.src.models.baseline_mlp import MLP


class RetrievalDeltaMLP(nn.Module):
    def __init__(self, query_dim: int, retrieved_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = MLP(query_dim + retrieved_dim, output_dim, hidden_dim=hidden_dim)

    def forward(self, query: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([query, retrieved], dim=-1))

