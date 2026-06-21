from __future__ import annotations

import numpy as np


def _parse_naca4(code: str) -> tuple[float, float, float]:
    digits = code.strip().upper().replace("NACA", "")
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"Expected a NACA 4-digit code, got {code!r}.")
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0
    return m, p, t


def generate_naca4(code: str, n_points: int = 200) -> dict:
    """Generate a NACA 4-digit airfoil on a cosine-spaced chord grid."""
    if n_points < 16:
        raise ValueError("n_points must be at least 16.")

    m, p, t = _parse_naca4(code)
    beta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if m > 0 and p > 0:
        left = x < p
        right = ~left
        yc[left] = m / p**2 * (2 * p * x[left] - x[left] ** 2)
        dyc_dx[left] = 2 * m / p**2 * (p - x[left])
        yc[right] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[right] - x[right] ** 2)
        dyc_dx[right] = 2 * m / (1 - p) ** 2 * (p - x[right])

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    closed_x = np.concatenate([xu[::-1], xl[1:]])
    closed_y = np.concatenate([yu[::-1], yl[1:]])

    return {
        "code": code,
        "x": x.astype(np.float32),
        "yu": yu.astype(np.float32),
        "yl": yl.astype(np.float32),
        "coords": np.column_stack([closed_x, closed_y]).astype(np.float32),
        "params": np.array([m, p, t], dtype=np.float32),
        "metadata": {"camber": m, "camber_position": p, "thickness": t},
    }


def generate_naca4_from_params(params: np.ndarray, n_points: int = 200) -> dict:
    """Generate a NACA-like 4-digit airfoil from continuous m, p, t parameters."""
    m, p, t = [float(v) for v in params]
    m = float(np.clip(m, 0.0, 0.09))
    p = float(np.clip(p, 0.1, 0.9))
    t = float(np.clip(t, 0.04, 0.24))
    code = f"{round(m * 100):1.0f}{round(p * 10):1.0f}{round(t * 100):02.0f}"
    airfoil = generate_naca4(code, n_points=n_points)
    airfoil["params"] = np.array([m, p, t], dtype=np.float32)
    airfoil["metadata"] = {"camber": m, "camber_position": p, "thickness": t}
    return airfoil

