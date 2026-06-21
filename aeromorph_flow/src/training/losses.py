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


def cp_delta_to_cl_delta(delta_cp: torch.Tensor) -> torch.Tensor:
    """Approximate delta Cl by integrating lower-minus-upper delta Cp over chord."""
    cp_dim = delta_cp.shape[1]
    if cp_dim % 2 != 0:
        raise ValueError("delta_cp must contain equal upper and lower surface grids.")
    n_points = cp_dim // 2
    beta = torch.linspace(0.0, torch.pi, n_points, device=delta_cp.device, dtype=delta_cp.dtype)
    x = 0.5 * (1.0 - torch.cos(beta))
    delta_cp_upper = delta_cp[:, :n_points]
    delta_cp_lower = delta_cp[:, n_points:]
    pressure_jump = delta_cp_lower - delta_cp_upper
    return torch.trapezoid(pressure_jump, x, dim=1)


def cp_cl_consistency_loss(pred: torch.Tensor, cp_dim: int) -> torch.Tensor:
    """Encourage the scalar delta Cl head to agree with integrated predicted delta Cp."""
    delta_cl_from_cp = cp_delta_to_cl_delta(pred[:, :cp_dim])
    delta_cl_head = pred[:, cp_dim]
    return torch.mean((delta_cl_head - delta_cl_from_cp) ** 2)
