from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from aeromorph_flow.src.models.delta_model import DeltaMLP
from aeromorph_flow.src.training.dataset import DeltaDataset, split_arrays_by_path
from aeromorph_flow.src.utils.io import load_npz


def _load_model(path: Path) -> DeltaMLP:
    checkpoint = torch.load(path, map_location="cpu")
    model = DeltaMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _predict_delta(model: torch.nn.Module, arrays: dict[str, np.ndarray]) -> np.ndarray:
    dataset = DeltaDataset(arrays)
    with torch.no_grad():
        return model(torch.from_numpy(dataset.x)).cpu().numpy()


def _group_indices_by_path(arrays: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    path_ids = arrays["path_id"].reshape(-1)
    groups = {}
    for path_id in np.unique(path_ids):
        idx = np.where(path_ids == path_id)[0]
        step_order = np.argsort(arrays["step_index"].reshape(-1)[idx])
        groups[int(path_id)] = idx[step_order]
    return groups


def evaluate_path_consistency(arrays: dict[str, np.ndarray], model: torch.nn.Module) -> list[dict]:
    pred_delta = _predict_delta(model, arrays)
    cp_dim = arrays["delta_cp"].shape[1]
    results = []
    for path_id, idx in _group_indices_by_path(arrays).items():
        if len(idx) < 2:
            continue

        pred_delta_cp = np.sum(pred_delta[idx, :cp_dim], axis=0)
        pred_delta_cl = float(np.sum(pred_delta[idx, cp_dim]))
        pred_delta_cd = float(np.sum(pred_delta[idx, cp_dim + 1]))

        actual_delta_cp = arrays["cp_after"][idx[-1]] - arrays["cp_before"][idx[0]]
        actual_delta_cl = float(arrays["cl_after"][idx[-1], 0] - arrays["cl_before"][idx[0], 0])
        actual_delta_cd = float(arrays["cd_after"][idx[-1], 0] - arrays["cd_before"][idx[0], 0])

        step_cp_mae = float(np.mean(np.abs(pred_delta[idx, :cp_dim] - arrays["delta_cp"][idx])))
        endpoint_cp_mae = float(np.mean(np.abs(pred_delta_cp - actual_delta_cp)))
        results.append(
            {
                "path_id": path_id,
                "n_steps": int(len(idx)),
                "step_cp_mae": step_cp_mae,
                "endpoint_cp_mae": endpoint_cp_mae,
                "endpoint_cl_abs_error": abs(pred_delta_cl - actual_delta_cl),
                "endpoint_cd_abs_error": abs(pred_delta_cd - actual_delta_cd),
                "actual_endpoint_delta_cl": actual_delta_cl,
                "pred_endpoint_delta_cl": pred_delta_cl,
                "actual_endpoint_delta_cd": actual_delta_cd,
                "pred_endpoint_delta_cd": pred_delta_cd,
            }
        )
    return results


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, data_path: Path, model_path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        endpoint_cp = np.array([row["endpoint_cp_mae"] for row in rows], dtype=float)
        endpoint_cl = np.array([row["endpoint_cl_abs_error"] for row in rows], dtype=float)
        endpoint_cd = np.array([row["endpoint_cd_abs_error"] for row in rows], dtype=float)
        step_cp = np.array([row["step_cp_mae"] for row in rows], dtype=float)
    else:
        endpoint_cp = endpoint_cl = endpoint_cd = step_cp = np.array([np.nan])

    lines = [
        "# Path-Consistency Evaluation",
        "",
        f"Dataset: `{data_path}`",
        f"Model: `{model_path}`",
        "",
        "Metric: sum predicted one-step deltas over a held-out morph path and compare to the actual endpoint delta.",
        "",
        "## Summary",
        "",
        f"- Paths evaluated: {len(rows)}",
        f"- Mean one-step Cp MAE: {float(np.nanmean(step_cp)):.6f}",
        f"- Mean endpoint Cp MAE: {float(np.nanmean(endpoint_cp)):.6f}",
        f"- Median endpoint Cp MAE: {float(np.nanmedian(endpoint_cp)):.6f}",
        f"- Mean endpoint Cl abs error: {float(np.nanmean(endpoint_cl)):.6f}",
        f"- Mean endpoint Cd abs error: {float(np.nanmean(endpoint_cd)):.6f}",
        "",
        "## Sample Paths",
        "",
        "| path | steps | endpoint Cp MAE | endpoint Cl err | endpoint Cd err | actual dCl | pred dCl | actual dCd | pred dCd |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:12]:
        lines.append(
            f"| {row['path_id']} | {row['n_steps']} | {row['endpoint_cp_mae']:.6f} | "
            f"{row['endpoint_cl_abs_error']:.6f} | {row['endpoint_cd_abs_error']:.6f} | "
            f"{row['actual_endpoint_delta_cl']:.6f} | {row['pred_endpoint_delta_cl']:.6f} | "
            f"{row['actual_endpoint_delta_cd']:.6f} | {row['pred_endpoint_delta_cd']:.6f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("aeromorph_flow/data/processed/xfoil_2000_hardened_transitions.npz"))
    parser.add_argument("--model", type=Path, default=Path("aeromorph_flow/data/processed/delta_mlp.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("aeromorph_flow/reports/path_consistency"))
    args = parser.parse_args()

    arrays = load_npz(args.data)
    _train_arrays, val_arrays = split_arrays_by_path(arrays, val_fraction=0.2, seed=8)
    model = _load_model(args.model)
    rows = evaluate_path_consistency(val_arrays, model)
    _write_csv(args.out_dir / "path_consistency.csv", rows)
    _write_markdown(args.out_dir / "path_consistency_report.md", args.data, args.model, rows)
    print(f"paths_evaluated={len(rows)}")
    if rows:
        print(f"mean_endpoint_cp_mae={np.mean([row['endpoint_cp_mae'] for row in rows]):.6f}")
        print(f"mean_endpoint_cl_abs_error={np.mean([row['endpoint_cl_abs_error'] for row in rows]):.6f}")
        print(f"mean_endpoint_cd_abs_error={np.mean([row['endpoint_cd_abs_error'] for row in rows]):.6f}")
    print(f"wrote_report={args.out_dir / 'path_consistency_report.md'}")
    print(f"wrote_csv={args.out_dir / 'path_consistency.csv'}")


if __name__ == "__main__":
    main()
