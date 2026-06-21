from __future__ import annotations

import numpy as np


def solve_airfoil_mock(airfoil: dict, aoa_deg: float, reynolds: float) -> dict:
    """Return plausible synthetic airfoil coefficients and Cp distribution."""
    x = np.asarray(airfoil["x"], dtype=np.float32)
    m, p, t = [float(v) for v in airfoil["params"]]
    aoa_rad = np.deg2rad(float(aoa_deg))
    log_re = np.log10(max(float(reynolds), 1.0))

    lift_slope = 2.0 * np.pi * (1.0 + 0.35 * m - 0.08 * abs(t - 0.12))
    cl = lift_slope * (aoa_rad + 1.8 * m * (0.5 + p)) * (1.0 + 0.015 * (log_re - 6.0))
    cd = 0.006 + 0.035 * cl**2 + 0.65 * (t - 0.12) ** 2 + 0.0007 * abs(aoa_deg)
    cm = -0.05 - 0.55 * m + 0.02 * (p - 0.4) - 0.01 * aoa_rad

    suction_peak = (1.2 + 0.12 * aoa_deg + 4.0 * m) * np.exp(-((x - 0.08) / 0.12) ** 2)
    recovery = 0.35 * (1.0 - x) * (1.0 + 2.0 * t)
    camber_wave = 0.4 * m * np.sin(np.pi * np.clip(x / max(p, 0.1), 0.0, 1.0))
    cp_upper = 1.0 - suction_peak - recovery - camber_wave
    cp_lower = 0.35 + 0.45 * aoa_rad + 0.6 * m * (1.0 - x) + 0.12 * t * np.cos(np.pi * x)
    cp = np.concatenate([cp_upper, cp_lower]).astype(np.float32)

    return {
        "cp": cp,
        "cl": float(cl),
        "cd": float(cd),
        "cm": float(cm),
        "converged": True,
    }

