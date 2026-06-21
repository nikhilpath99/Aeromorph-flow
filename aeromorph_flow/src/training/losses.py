from __future__ import annotations

import torch
import torch.nn.functional as F


def cd_nonnegative_penalty(cd_after_pred: torch.Tensor) -> torch.Tensor:
    """Soft penalty for violating Cd >= 0."""
    return torch.mean(F.relu(-cd_after_pred) ** 2)


def prediction_loss_with_cd_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    cd_index: int,
    cd_penalty_weight: float = 0.0,
    cd_after_pred: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = torch.mean((pred - target) ** 2)
    if cd_penalty_weight > 0.0:
        cd_values = pred[:, cd_index] if cd_after_pred is None else cd_after_pred.reshape(-1)
        loss = loss + cd_penalty_weight * cd_nonnegative_penalty(cd_values)
    return loss
