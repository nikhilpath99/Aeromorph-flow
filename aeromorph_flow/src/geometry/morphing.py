from __future__ import annotations

import numpy as np

from aeromorph_flow.src.geometry.naca import generate_naca4_from_params


def morph_airfoils(airfoil_a: dict, airfoil_b: dict, n_steps: int) -> list[dict]:
    """Linearly interpolate between two compatible airfoil parameter vectors."""
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2.")

    params_a = np.asarray(airfoil_a["params"], dtype=np.float32)
    params_b = np.asarray(airfoil_b["params"], dtype=np.float32)
    n_points = len(airfoil_a["x"])
    path = []
    for alpha in np.linspace(0.0, 1.0, n_steps):
        params = (1.0 - alpha) * params_a + alpha * params_b
        path.append(generate_naca4_from_params(params, n_points=n_points))
    return path


def surface_feature(airfoil: dict) -> np.ndarray:
    """Compact fixed-grid geometry feature: upper and lower y coordinates."""
    return np.concatenate([airfoil["yu"], airfoil["yl"]]).astype(np.float32)

