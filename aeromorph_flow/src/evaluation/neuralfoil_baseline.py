from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from aeromorph_flow.src.models.baseline_mlp import MLP
from aeromorph_flow.src.models.delta_model import DeltaMLP
from aeromorph_flow.src.training.dataset import BaselineDataset, DeltaDataset, split_arrays_by_path
from aeromorph_flow.src.utils.io import load_npz


def _surface_coordinates(surface_feature: np.ndarray) -> np.ndarray:
    n_points = surface_feature.shape[0] // 2
    beta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    yu = surface_feature[:n_points]
    yl = surface_feature[n_points:]
    return np.column_stack([np.r_[x[::-1], x[1:]], np.r_[yu[::-1], yl[1:]]])


def _as_float(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def _load_baseline(path: Path) -> MLP:
    checkpoint = torch.load(path, map_location="cpu")
    model = MLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=128,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _load_delta(path: Path) -> DeltaMLP:
    checkpoint = torch.load(path, map_location="cpu")
    model = DeltaMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=128,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _model_predictions(
    val_arrays: dict[str, np.ndarray],
    baseline_checkpoint: Path,
    delta_checkpoint: Path,
) -> dict[str, np.ndarray]:
    baseline_ds = BaselineDataset(val_arrays)
    delta_ds = DeltaDataset(val_arrays)
    cp_dim = val_arrays["cp_after"].shape[1]
    baseline = _load_baseline(baseline_checkpoint)
    delta = _load_delta(delta_checkpoint)
    with torch.no_grad():
        baseline_pred = baseline(torch.from_numpy(baseline_ds.x)).numpy()
        delta_pred = delta(torch.from_numpy(delta_ds.x)).numpy()

    return {
        "baseline_cl": baseline_pred[:, cp_dim],
        "baseline_cd": baseline_pred[:, cp_dim + 1],
        "delta_cl": val_arrays["cl_before"].reshape(-1) + delta_pred[:, cp_dim],
        "delta_cd": val_arrays["cd_before"].reshape(-1) + delta_pred[:, cp_dim + 1],
    }


def _evaluate_neuralfoil(
    val_arrays: dict[str, np.ndarray],
    model_size: str,
    n_crit: float,
) -> dict[str, np.ndarray]:
    try:
        import neuralfoil as nf
    except ImportError as exc:
        raise RuntimeError("NeuralFoil is not installed. Run: python -m pip install NeuralFoil") from exc

    cl_pred = []
    cd_pred = []
    confidence = []
    for i in range(len(val_arrays["aoa"])):
        coordinates = _surface_coordinates(val_arrays["geometry_after"][i])
        output = nf.get_aero_from_coordinates(
            coordinates=coordinates,
            alpha=_as_float(val_arrays["aoa"][i]),
            Re=_as_float(val_arrays["reynolds"][i]),
            n_crit=n_crit,
            model_size=model_size,
        )
        cl_pred.append(_as_float(output["CL"]))
        cd_pred.append(_as_float(output["CD"]))
        confidence.append(_as_float(output["analysis_confidence"]))
        if (i + 1) % 250 == 0:
            print(f"neuralfoil_evaluated={i + 1}")

    return {
        "neuralfoil_cl": np.asarray(cl_pred, dtype=np.float32),
        "neuralfoil_cd": np.asarray(cd_pred, dtype=np.float32),
        "neuralfoil_confidence": np.asarray(confidence, dtype=np.float32),
    }


def _write_predictions(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("aeromorph_flow/data/processed/xfoil_10k_transitions.npz"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("aeromorph_flow/data/processed/baseline_mlp_xfoil_10k.pt"))
    parser.add_argument("--delta-checkpoint", type=Path, default=Path("aeromorph_flow/data/processed/delta_mlp_xfoil_10k_unweighted.pt"))
    parser.add_argument("--out", type=Path, default=Path("aeromorph_flow/reports/neuralfoil_10k_comparison.csv"))
    parser.add_argument("--model-size", type=str, default="xlarge")
    parser.add_argument("--n-crit", type=float, default=9.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    arrays = load_npz(args.data)
    _train_arrays, val_arrays = split_arrays_by_path(arrays, val_fraction=0.2, seed=args.seed)
    if args.max_samples is not None:
        val_arrays = {key: value[: args.max_samples] for key, value in val_arrays.items()}

    model_preds = _model_predictions(val_arrays, args.baseline_checkpoint, args.delta_checkpoint)
    neuralfoil_preds = _evaluate_neuralfoil(val_arrays, model_size=args.model_size, n_crit=args.n_crit)

    truth_cl = val_arrays["cl_after"].reshape(-1)
    truth_cd = val_arrays["cd_after"].reshape(-1)
    rows = []
    for i in range(len(truth_cl)):
        rows.append(
            {
                "index": i,
                "path_id": int(val_arrays["path_id"][i].reshape(-1)[0]),
                "step_index": int(val_arrays["step_index"][i].reshape(-1)[0]),
                "aoa": _as_float(val_arrays["aoa"][i]),
                "reynolds": _as_float(val_arrays["reynolds"][i]),
                "truth_cl": float(truth_cl[i]),
                "truth_cd": float(truth_cd[i]),
                "neuralfoil_cl": float(neuralfoil_preds["neuralfoil_cl"][i]),
                "neuralfoil_cd": float(neuralfoil_preds["neuralfoil_cd"][i]),
                "neuralfoil_confidence": float(neuralfoil_preds["neuralfoil_confidence"][i]),
                "baseline_cl": float(model_preds["baseline_cl"][i]),
                "baseline_cd": float(model_preds["baseline_cd"][i]),
                "delta_cl": float(model_preds["delta_cl"][i]),
                "delta_cd": float(model_preds["delta_cd"][i]),
            }
        )

    _write_predictions(args.out, rows)
    print(f"n={len(rows)}")
    for name, cl_pred, cd_pred in [
        ("neuralfoil", neuralfoil_preds["neuralfoil_cl"], neuralfoil_preds["neuralfoil_cd"]),
        ("baseline", model_preds["baseline_cl"], model_preds["baseline_cd"]),
        ("delta", model_preds["delta_cl"], model_preds["delta_cd"]),
    ]:
        print(
            f"{name}: cl_mae={_mae(cl_pred, truth_cl):.6f} "
            f"cd_mae={_mae(cd_pred, truth_cd):.6f} "
            f"cd_drag_counts={_mae(cd_pred, truth_cd) / 1e-4:.2f}"
        )
    print(
        "neuralfoil_confidence: "
        f"mean={float(np.mean(neuralfoil_preds['neuralfoil_confidence'])):.6f} "
        f"min={float(np.min(neuralfoil_preds['neuralfoil_confidence'])):.6f}"
    )
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
