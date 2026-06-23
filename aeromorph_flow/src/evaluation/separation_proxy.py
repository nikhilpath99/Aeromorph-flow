from __future__ import annotations

import numpy as np


def cp_separation_proxy(cp: np.ndarray) -> dict[str, np.ndarray]:
    """Compute simple upper-surface pressure-recovery proxies from Cp arrays."""
    cp = np.asarray(cp, dtype=np.float32)
    if cp.ndim == 1:
        cp = cp[None, :]
    cp_dim = cp.shape[1]
    if cp_dim % 2 != 0:
        raise ValueError("Cp feature must contain equal upper and lower surface grids.")

    n_points = cp_dim // 2
    beta = np.linspace(0.0, np.pi, n_points, dtype=np.float32)
    x = 0.5 * (1.0 - np.cos(beta))
    upper = cp[:, :n_points]
    lower = cp[:, n_points:]

    min_idx = np.argmin(upper, axis=1)
    row = np.arange(upper.shape[0])
    cp_min = upper[row, min_idx]
    x_cp_min = x[min_idx]
    cp_te = upper[:, -1]
    recovery = cp_te - cp_min
    recovery_length = np.maximum(1.0 - x_cp_min, 1e-4)
    recovery_gradient = recovery / recovery_length
    dcpdx = np.gradient(upper, x, axis=1, edge_order=1)

    positive_recovery_gradient = np.zeros_like(recovery_gradient)
    max_positive_dcpdx_after_min = np.zeros_like(recovery_gradient)
    for i, idx in enumerate(min_idx):
        after_min = dcpdx[i, idx:]
        positive = np.maximum(after_min, 0.0)
        positive_recovery_gradient[i] = float(np.mean(positive))
        max_positive_dcpdx_after_min[i] = float(np.max(positive))

    return {
        "cp_min_upper": cp_min.astype(np.float32),
        "x_cp_min_upper": x_cp_min.astype(np.float32),
        "cp_recovery_upper": recovery.astype(np.float32),
        "cp_recovery_gradient_upper": recovery_gradient.astype(np.float32),
        "mean_positive_dcpdx_upper_after_min": positive_recovery_gradient.astype(np.float32),
        "max_positive_dcpdx_upper_after_min": max_positive_dcpdx_after_min.astype(np.float32),
        "cp_pressure_jump_integral": np.trapezoid(lower - upper, x, axis=1).astype(np.float32),
    }


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
