from __future__ import annotations

import numpy as np

from aeromorph_flow.src.geometry.morphing import surface_feature


def transition_features(airfoil_before: dict, airfoil_after: dict) -> dict:
    geom_before = surface_feature(airfoil_before)
    geom_after = surface_feature(airfoil_after)
    return {
        "geometry_before": geom_before,
        "geometry_after": geom_after,
        "delta_geometry": (geom_after - geom_before).astype(np.float32),
        "params_before": airfoil_before["params"].astype(np.float32),
        "params_after": airfoil_after["params"].astype(np.float32),
        "delta_params": (airfoil_after["params"] - airfoil_before["params"]).astype(np.float32),
    }

