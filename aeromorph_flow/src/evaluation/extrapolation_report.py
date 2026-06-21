from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aeromorph_flow.src.models.baseline_mlp import MLP
from aeromorph_flow.src.models.delta_model import DeltaMLP
from aeromorph_flow.src.training.dataset import (
    BaselineDataset,
    DeltaDataset,
    split_arrays_by_path,
    split_arrays_extrapolation,
)
from aeromorph_flow.src.utils.io import load_npz


@dataclass
class ModelResult:
    model_name: str
    split_name: str
    train_n: int
    val_n: int
    best_epoch: int
    best_val_mse: float
    cp_mae: float
    cl_mae: float
    cd_mae: float
    predictions: dict[str, np.ndarray]


def _make_split(arrays: dict[str, np.ndarray], split_name: str):
    if split_name == "path":
        return split_arrays_by_path(arrays, val_fraction=0.2, seed=8)
    return split_arrays_extrapolation(arrays, mode=split_name)


def _train_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_epoch = 0
    best_mse = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in val_loader:
                err = model(x) - y
                losses.append(torch.mean(err**2).item())
        val_mse = float(np.mean(losses))
        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, best_mse


def _predict(model: torch.nn.Module, dataset, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for x, _y in loader:
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _baseline_result(
    arrays: dict[str, np.ndarray],
    split_name: str,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    hidden_dim: int,
) -> ModelResult:
    train_ds = BaselineDataset(train_arrays)
    val_ds = BaselineDataset(val_arrays)
    model = MLP(train_ds.x.shape[1], train_ds.y.shape[1], hidden_dim=hidden_dim)
    best_epoch, best_mse = _train_model(
        model,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        epochs=epochs,
    )
    pred = _predict(model, val_ds, batch_size=batch_size)
    cp_dim = val_arrays["cp_after"].shape[1]
    cp_pred = pred[:, :cp_dim]
    cl_pred = pred[:, cp_dim : cp_dim + 1]
    cd_pred = pred[:, cp_dim + 1 : cp_dim + 2]
    cp_actual = val_arrays["cp_after"]
    cl_actual = val_arrays["cl_after"]
    cd_actual = val_arrays["cd_after"]
    return ModelResult(
        model_name="absolute_baseline",
        split_name=split_name,
        train_n=len(train_ds),
        val_n=len(val_ds),
        best_epoch=best_epoch,
        best_val_mse=best_mse,
        cp_mae=float(np.mean(np.abs(cp_pred - cp_actual))),
        cl_mae=float(np.mean(np.abs(cl_pred - cl_actual))),
        cd_mae=float(np.mean(np.abs(cd_pred - cd_actual))),
        predictions={
            "path_id": val_arrays["path_id"].reshape(-1),
            "step_index": val_arrays["step_index"].reshape(-1),
            "actual_cl": cl_actual.reshape(-1),
            "actual_cd": cd_actual.reshape(-1),
            "pred_cl": cl_pred.reshape(-1),
            "pred_cd": cd_pred.reshape(-1),
            "actual_delta_cl": val_arrays["delta_cl"].reshape(-1),
            "actual_delta_cd": val_arrays["delta_cd"].reshape(-1),
        },
    )


def _delta_result(
    arrays: dict[str, np.ndarray],
    split_name: str,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    hidden_dim: int,
) -> ModelResult:
    train_ds = DeltaDataset(train_arrays)
    val_ds = DeltaDataset(val_arrays)
    model = DeltaMLP(train_ds.x.shape[1], train_ds.y.shape[1], hidden_dim=hidden_dim)
    best_epoch, best_mse = _train_model(
        model,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        epochs=epochs,
    )
    pred_delta = _predict(model, val_ds, batch_size=batch_size)
    cp_dim = val_arrays["delta_cp"].shape[1]
    cp_pred = val_arrays["cp_before"] + pred_delta[:, :cp_dim]
    cl_pred = val_arrays["cl_before"] + pred_delta[:, cp_dim : cp_dim + 1]
    cd_pred = val_arrays["cd_before"] + pred_delta[:, cp_dim + 1 : cp_dim + 2]
    cp_actual = val_arrays["cp_after"]
    cl_actual = val_arrays["cl_after"]
    cd_actual = val_arrays["cd_after"]
    return ModelResult(
        model_name="delta_model",
        split_name=split_name,
        train_n=len(train_ds),
        val_n=len(val_ds),
        best_epoch=best_epoch,
        best_val_mse=best_mse,
        cp_mae=float(np.mean(np.abs(cp_pred - cp_actual))),
        cl_mae=float(np.mean(np.abs(cl_pred - cl_actual))),
        cd_mae=float(np.mean(np.abs(cd_pred - cd_actual))),
        predictions={
            "path_id": val_arrays["path_id"].reshape(-1),
            "step_index": val_arrays["step_index"].reshape(-1),
            "actual_cl": cl_actual.reshape(-1),
            "actual_cd": cd_actual.reshape(-1),
            "pred_cl": cl_pred.reshape(-1),
            "pred_cd": cd_pred.reshape(-1),
            "actual_delta_cl": val_arrays["delta_cl"].reshape(-1),
            "actual_delta_cd": val_arrays["delta_cd"].reshape(-1),
        },
    )


def _write_metrics(path: Path, results: list[ModelResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "model",
                "train_n",
                "val_n",
                "best_epoch",
                "best_val_mse",
                "cp_mae",
                "cl_mae",
                "cd_mae",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "split": result.split_name,
                    "model": result.model_name,
                    "train_n": result.train_n,
                    "val_n": result.val_n,
                    "best_epoch": result.best_epoch,
                    "best_val_mse": result.best_val_mse,
                    "cp_mae": result.cp_mae,
                    "cl_mae": result.cl_mae,
                    "cd_mae": result.cd_mae,
                }
            )


def _write_predictions(path: Path, results: list[ModelResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "model",
                "path_id",
                "step_index",
                "lift_coefficient_cl_actual",
                "drag_coefficient_cd_actual",
                "lift_coefficient_cl_pred",
                "drag_coefficient_cd_pred",
                "delta_cl_actual",
                "delta_cd_actual",
                "cl_abs_error",
                "cd_abs_error",
            ],
        )
        writer.writeheader()
        for result in results:
            pred = result.predictions
            for i in range(len(pred["actual_cl"])):
                actual_cl = float(pred["actual_cl"][i])
                actual_cd = float(pred["actual_cd"][i])
                pred_cl = float(pred["pred_cl"][i])
                pred_cd = float(pred["pred_cd"][i])
                writer.writerow(
                    {
                        "split": result.split_name,
                        "model": result.model_name,
                        "path_id": int(pred["path_id"][i]),
                        "step_index": int(pred["step_index"][i]),
                        "lift_coefficient_cl_actual": actual_cl,
                        "drag_coefficient_cd_actual": actual_cd,
                        "lift_coefficient_cl_pred": pred_cl,
                        "drag_coefficient_cd_pred": pred_cd,
                        "delta_cl_actual": float(pred["actual_delta_cl"][i]),
                        "delta_cd_actual": float(pred["actual_delta_cd"][i]),
                        "cl_abs_error": abs(pred_cl - actual_cl),
                        "cd_abs_error": abs(pred_cd - actual_cd),
                    }
                )


def _write_markdown(path: Path, dataset_path: Path, results: list[ModelResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AeroMorph-Flow Extrapolation Report",
        "",
        f"Dataset: `{dataset_path}`",
        "",
        "Lift is reported as the lift coefficient `Cl`; dimensional lift in Newtons needs velocity, density, and reference area.",
        "",
        "## Metrics",
        "",
        "| Split | Model | Train n | Val n | Best epoch | Val MSE | Cp MAE | Cl MAE | Cd MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.split_name} | {result.model_name} | {result.train_n} | {result.val_n} | "
            f"{result.best_epoch} | {result.best_val_mse:.6f} | {result.cp_mae:.6f} | "
            f"{result.cl_mae:.6f} | {result.cd_mae:.6f} |"
        )

    lines.extend(["", "## Lift/Drag Samples", ""])
    for result in results:
        pred = result.predictions
        lines.append(f"### {result.split_name} / {result.model_name}")
        lines.append("")
        lines.append("| path | step | actual Cl | pred Cl | actual Cd | pred Cd |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for i in range(min(8, len(pred["actual_cl"]))):
            lines.append(
                f"| {int(pred['path_id'][i])} | {int(pred['step_index'][i])} | "
                f"{float(pred['actual_cl'][i]):.5f} | {float(pred['pred_cl'][i]):.5f} | "
                f"{float(pred['actual_cd'][i]):.5f} | {float(pred['pred_cd'][i]):.5f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("aeromorph_flow/data/processed/xfoil_2000_transitions.npz"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["ood_thickness_high", "ood_camber_high", "ood_morph_large"],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("aeromorph_flow/reports"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arrays = load_npz(args.data)
    results: list[ModelResult] = []
    for split_name in args.splits:
        train_arrays, val_arrays = _make_split(arrays, split_name)
        print(f"split={split_name} train_n={len(next(iter(train_arrays.values())))} val_n={len(next(iter(val_arrays.values())))}")
        results.append(
            _baseline_result(arrays, split_name, train_arrays, val_arrays, args.epochs, args.batch_size, args.hidden_dim)
        )
        print(
            f"{split_name} baseline cp_mae={results[-1].cp_mae:.6f} "
            f"cl_mae={results[-1].cl_mae:.6f} cd_mae={results[-1].cd_mae:.6f}"
        )
        results.append(
            _delta_result(arrays, split_name, train_arrays, val_arrays, args.epochs, args.batch_size, args.hidden_dim)
        )
        print(
            f"{split_name} delta cp_mae={results[-1].cp_mae:.6f} "
            f"cl_mae={results[-1].cl_mae:.6f} cd_mae={results[-1].cd_mae:.6f}"
        )

    _write_metrics(args.out_dir / "extrapolation_metrics.csv", results)
    _write_predictions(args.out_dir / "extrapolation_lift_drag_predictions.csv", results)
    _write_markdown(args.out_dir / "extrapolation_report.md", args.data, results)
    print(f"wrote_report={args.out_dir / 'extrapolation_report.md'}")
    print(f"wrote_metrics={args.out_dir / 'extrapolation_metrics.csv'}")
    print(f"wrote_predictions={args.out_dir / 'extrapolation_lift_drag_predictions.csv'}")


if __name__ == "__main__":
    main()
