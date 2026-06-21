from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from aeromorph_flow.src.geometry.features import transition_features
from aeromorph_flow.src.geometry.morphing import morph_airfoils
from aeromorph_flow.src.geometry.naca import generate_naca4_from_params
from aeromorph_flow.src.solvers.mock_solver import solve_airfoil_mock


@dataclass(frozen=True)
class DatasetConfig:
    n_paths: int = 64
    n_steps: int = 5
    n_points: int = 96
    aoa_min: float = -2.0
    aoa_max: float = 8.0
    re_min: float = 5e5
    re_max: float = 2e6
    seed: int = 7
    solver: str = "mock"
    xfoil_path: str | None = None
    xfoil_timeout_s: float = 30.0


def random_naca_params(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [
            rng.uniform(0.00, 0.06),
            rng.uniform(0.20, 0.60),
            rng.uniform(0.08, 0.18),
        ],
        dtype=np.float32,
    )


def generate_transition_dataset(config: DatasetConfig) -> dict[str, np.ndarray]:
    if config.solver not in {"mock", "xfoil"}:
        raise ValueError(f"Unknown solver {config.solver!r}; expected 'mock' or 'xfoil'.")

    if config.solver == "xfoil":
        from aeromorph_flow.src.solvers.xfoil_runner import find_xfoil_executable, solve_airfoil_xfoil

        find_xfoil_executable(config.xfoil_path)

    rng = np.random.default_rng(config.seed)
    rows = []
    for path_id in range(config.n_paths):
        a = generate_naca4_from_params(random_naca_params(rng), n_points=config.n_points)
        b = generate_naca4_from_params(random_naca_params(rng), n_points=config.n_points)
        path = morph_airfoils(a, b, n_steps=config.n_steps)
        aoa = float(rng.uniform(config.aoa_min, config.aoa_max))
        reynolds = float(np.exp(rng.uniform(np.log(config.re_min), np.log(config.re_max))))
        flows = []
        for airfoil in path:
            try:
                if config.solver == "xfoil":
                    flow = solve_airfoil_xfoil(
                        airfoil,
                        aoa,
                        reynolds,
                        xfoil_path=config.xfoil_path,
                        timeout_s=config.xfoil_timeout_s,
                    )
                else:
                    flow = solve_airfoil_mock(airfoil, aoa, reynolds)
            except Exception:
                flow = {"converged": False}
            flows.append(flow)

        for step_index, (before, after) in enumerate(zip(path[:-1], path[1:])):
            flow_before = flows[step_index]
            flow_after = flows[step_index + 1]
            if not flow_before.get("converged", False) or not flow_after.get("converged", False):
                continue
            feats = transition_features(before, after)
            rows.append(
                {
                    **feats,
                    "aoa": np.array([aoa], dtype=np.float32),
                    "log_re": np.array([np.log10(reynolds)], dtype=np.float32),
                    "cp_before": flow_before["cp"],
                    "cp_after": flow_after["cp"],
                    "delta_cp": flow_after["cp"] - flow_before["cp"],
                    "cl_before": np.array([flow_before["cl"]], dtype=np.float32),
                    "cl_after": np.array([flow_after["cl"]], dtype=np.float32),
                    "delta_cl": np.array([flow_after["cl"] - flow_before["cl"]], dtype=np.float32),
                    "cd_before": np.array([flow_before["cd"]], dtype=np.float32),
                    "cd_after": np.array([flow_after["cd"]], dtype=np.float32),
                    "delta_cd": np.array([flow_after["cd"] - flow_before["cd"]], dtype=np.float32),
                    "path_id": np.array([path_id], dtype=np.int64),
                    "step_index": np.array([step_index], dtype=np.int64),
                }
            )

    if not rows:
        raise RuntimeError(f"No converged transition samples were generated with solver={config.solver!r}.")

    return {key: np.stack([row[key] for row in rows]) for key in rows[0].keys()}


class DeltaDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        self.x = np.concatenate(
            [
                arrays["geometry_before"],
                arrays["delta_geometry"],
                arrays["cp_before"],
                arrays["cl_before"],
                arrays["cd_before"],
                arrays["aoa"],
                arrays["log_re"],
            ],
            axis=1,
        ).astype(np.float32)
        self.y = np.concatenate(
            [arrays["delta_cp"], arrays["delta_cl"], arrays["delta_cd"]],
            axis=1,
        ).astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])


class BaselineDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        self.x = np.concatenate(
            [
                arrays["geometry_after"],
                arrays["aoa"],
                arrays["log_re"],
            ],
            axis=1,
        ).astype(np.float32)
        self.y = np.concatenate(
            [arrays["cp_after"], arrays["cl_after"], arrays["cd_after"]],
            axis=1,
        ).astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])


def split_arrays(arrays: dict[str, np.ndarray], val_fraction: float = 0.2, seed: int = 11):
    rng = np.random.default_rng(seed)
    n = len(next(iter(arrays.values())))
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train = {key: value[train_idx] for key, value in arrays.items()}
    val = {key: value[val_idx] for key, value in arrays.items()}
    return train, val


def split_arrays_by_path(arrays: dict[str, np.ndarray], val_fraction: float = 0.2, seed: int = 11):
    if "path_id" not in arrays:
        return split_arrays(arrays, val_fraction=val_fraction, seed=seed)

    rng = np.random.default_rng(seed)
    path_ids = arrays["path_id"].reshape(-1)
    unique_paths = np.unique(path_ids)
    shuffled_paths = rng.permutation(unique_paths)
    n_val_paths = max(1, int(len(shuffled_paths) * val_fraction))
    val_paths = set(int(path_id) for path_id in shuffled_paths[:n_val_paths])
    val_mask = np.array([int(path_id) in val_paths for path_id in path_ids], dtype=bool)
    train_mask = ~val_mask
    if not np.any(train_mask) or not np.any(val_mask):
        return split_arrays(arrays, val_fraction=val_fraction, seed=seed)
    train = {key: value[train_mask] for key, value in arrays.items()}
    val = {key: value[val_mask] for key, value in arrays.items()}
    return train, val


def split_arrays_extrapolation(arrays: dict[str, np.ndarray], mode: str):
    """Create fixed OOD splits with a gap between train and validation regimes."""
    if mode == "ood_thickness_high":
        before = arrays["params_before"][:, 2]
        after = arrays["params_after"][:, 2]
        train_mask = np.maximum(before, after) <= 0.14
        val_mask = np.minimum(before, after) >= 0.15
    elif mode == "ood_camber_high":
        before = arrays["params_before"][:, 0]
        after = arrays["params_after"][:, 0]
        train_mask = np.maximum(before, after) <= 0.035
        val_mask = np.minimum(before, after) >= 0.045
    elif mode == "ood_morph_large":
        morph_norm = np.linalg.norm(arrays["delta_params"], axis=1)
        train_limit = float(np.quantile(morph_norm, 0.70))
        val_limit = float(np.quantile(morph_norm, 0.90))
        train_mask = morph_norm <= train_limit
        val_mask = morph_norm >= val_limit
    else:
        raise ValueError(f"Unknown extrapolation split mode: {mode}")

    if not np.any(train_mask) or not np.any(val_mask):
        raise RuntimeError(
            f"Split {mode!r} produced train_n={int(np.sum(train_mask))}, "
            f"val_n={int(np.sum(val_mask))}."
        )
    train = {key: value[train_mask] for key, value in arrays.items()}
    val = {key: value[val_mask] for key, value in arrays.items()}
    return train, val
