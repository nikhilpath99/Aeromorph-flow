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
from aeromorph_flow.src.training.train_delta import weighted_prediction_loss
from aeromorph_flow.src.utils.io import load_npz
from aeromorph_flow.src.evaluation.separation_proxy import cp_separation_proxy, pearson_corr


SPLITS = [
    "path",
    "ood_thickness_high",
    "ood_camber_high",
    "ood_morph_large",
    "ood_aoa_high",
    "ood_re_high",
]


@dataclass
class EvalResult:
    split: str
    model: str
    train_n: int
    val_n: int
    cp_mae: float | None
    cl_mae: float
    cd_mae: float
    neuralfoil_confidence_mean: float | None = None
    neuralfoil_confidence_min: float | None = None


def _make_split(arrays: dict[str, np.ndarray], split: str):
    if split == "path":
        return split_arrays_by_path(arrays, val_fraction=0.2, seed=8)
    return split_arrays_extrapolation(arrays, mode=split)


def _as_float(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _surface_coordinates(surface_feature: np.ndarray) -> np.ndarray:
    n_points = surface_feature.shape[0] // 2
    beta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    yu = surface_feature[:n_points]
    yl = surface_feature[n_points:]
    return np.column_stack([np.r_[x[::-1], x[1:]], np.r_[yu[::-1], yl[1:]]])


def _prediction_metrics(cp_pred, cl_pred, cd_pred, val_arrays: dict[str, np.ndarray]) -> tuple[float, float, float]:
    return (
        float(np.mean(np.abs(cp_pred - val_arrays["cp_after"]))),
        float(np.mean(np.abs(cl_pred.reshape(-1, 1) - val_arrays["cl_after"]))),
        float(np.mean(np.abs(cd_pred.reshape(-1, 1) - val_arrays["cd_after"]))),
    )


def _train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    cp_dim: int,
    epochs: int,
    cl_weight: float = 1.0,
    cd_weight: float = 1.0,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            pred = model(x)
            loss = weighted_prediction_loss(
                pred,
                y,
                cp_dim=cp_dim,
                cl_weight=cl_weight,
                cd_weight=cd_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _eval_baseline(
    split: str,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    hidden_dim: int,
) -> tuple[EvalResult, list[dict]]:
    train_ds = BaselineDataset(train_arrays)
    val_ds = BaselineDataset(val_arrays)
    cp_dim = val_arrays["cp_after"].shape[1]
    model = MLP(train_ds.x.shape[1], train_ds.y.shape[1], hidden_dim=hidden_dim)
    _train_model(model, DataLoader(train_ds, batch_size=batch_size, shuffle=True), cp_dim, epochs)
    with torch.no_grad():
        pred = model(torch.from_numpy(val_ds.x)).numpy()
    cp_pred = pred[:, :cp_dim]
    cl_pred = pred[:, cp_dim]
    cd_pred = pred[:, cp_dim + 1]
    cp_mae, cl_mae, cd_mae = _prediction_metrics(cp_pred, cl_pred, cd_pred, val_arrays)
    rows = _prediction_rows(split, "baseline", val_arrays, cl_pred, cd_pred)
    return EvalResult(split, "baseline", len(train_ds), len(val_ds), cp_mae, cl_mae, cd_mae), rows


def _eval_delta(
    split: str,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    cd_weight: float,
    model_name: str,
) -> tuple[EvalResult, list[dict]]:
    train_ds = DeltaDataset(train_arrays)
    val_ds = DeltaDataset(val_arrays)
    cp_dim = val_arrays["delta_cp"].shape[1]
    model = DeltaMLP(train_ds.x.shape[1], train_ds.y.shape[1], hidden_dim=hidden_dim)
    _train_model(
        model,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        cp_dim,
        epochs,
        cd_weight=cd_weight,
    )
    with torch.no_grad():
        pred_delta = model(torch.from_numpy(val_ds.x)).numpy()
    cp_pred = val_arrays["cp_before"] + pred_delta[:, :cp_dim]
    cl_pred = val_arrays["cl_before"].reshape(-1) + pred_delta[:, cp_dim]
    cd_pred = val_arrays["cd_before"].reshape(-1) + pred_delta[:, cp_dim + 1]
    cp_mae, cl_mae, cd_mae = _prediction_metrics(cp_pred, cl_pred, cd_pred, val_arrays)
    rows = _prediction_rows(split, model_name, val_arrays, cl_pred, cd_pred)
    return EvalResult(split, model_name, len(train_ds), len(val_ds), cp_mae, cl_mae, cd_mae), rows


def _eval_neuralfoil(split: str, train_n: int, val_arrays: dict[str, np.ndarray], model_size: str, n_crit: float):
    try:
        import neuralfoil as nf
    except ImportError as exc:
        raise RuntimeError("NeuralFoil is not installed. Run: python -m pip install NeuralFoil") from exc

    cl_pred = []
    cd_pred = []
    confidence = []
    for i in range(len(val_arrays["aoa"])):
        output = nf.get_aero_from_coordinates(
            coordinates=_surface_coordinates(val_arrays["geometry_after"][i]),
            alpha=_as_float(val_arrays["aoa"][i]),
            Re=_as_float(val_arrays["reynolds"][i]),
            n_crit=n_crit,
            model_size=model_size,
        )
        cl_pred.append(_as_float(output["CL"]))
        cd_pred.append(_as_float(output["CD"]))
        confidence.append(_as_float(output["analysis_confidence"]))
    cl_pred_arr = np.asarray(cl_pred, dtype=np.float32)
    cd_pred_arr = np.asarray(cd_pred, dtype=np.float32)
    cl_mae = float(np.mean(np.abs(cl_pred_arr.reshape(-1, 1) - val_arrays["cl_after"])))
    cd_mae = float(np.mean(np.abs(cd_pred_arr.reshape(-1, 1) - val_arrays["cd_after"])))
    confidence_arr = np.asarray(confidence, dtype=np.float32)
    result = EvalResult(
        split=split,
        model="neuralfoil",
        train_n=train_n,
        val_n=len(val_arrays["aoa"]),
        cp_mae=None,
        cl_mae=cl_mae,
        cd_mae=cd_mae,
        neuralfoil_confidence_mean=float(np.mean(confidence_arr)),
        neuralfoil_confidence_min=float(np.min(confidence_arr)),
    )
    rows = _prediction_rows(split, "neuralfoil", val_arrays, cl_pred_arr, cd_pred_arr)
    return result, rows


def _prediction_rows(split: str, model: str, val_arrays: dict[str, np.ndarray], cl_pred, cd_pred) -> list[dict]:
    rows = []
    truth_cl = val_arrays["cl_after"].reshape(-1)
    truth_cd = val_arrays["cd_after"].reshape(-1)
    sep_before = cp_separation_proxy(val_arrays["cp_before"])
    sep_after = cp_separation_proxy(val_arrays["cp_after"])
    for i in range(len(truth_cl)):
        rows.append(
            {
                "split": split,
                "model": model,
                "path_id": int(val_arrays["path_id"][i].reshape(-1)[0]),
                "step_index": int(val_arrays["step_index"][i].reshape(-1)[0]),
                "truth_cl": float(truth_cl[i]),
                "truth_cd": float(truth_cd[i]),
                "pred_cl": float(cl_pred[i]),
                "pred_cd": float(cd_pred[i]),
                "cl_abs_error": float(abs(cl_pred[i] - truth_cl[i])),
                "cd_abs_error": float(abs(cd_pred[i] - truth_cd[i])),
                "cp_recovery_gradient_before": float(sep_before["cp_recovery_gradient_upper"][i]),
                "cp_recovery_gradient_after": float(sep_after["cp_recovery_gradient_upper"][i]),
                "max_positive_dcpdx_after": float(sep_after["max_positive_dcpdx_upper_after_min"][i]),
                "cp_min_upper_after": float(sep_after["cp_min_upper"][i]),
                "x_cp_min_upper_after": float(sep_after["x_cp_min_upper"][i]),
            }
        )
    return rows


def _write_metrics(path: Path, results: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path_cd = {result.model: result.cd_mae for result in results if result.split == "path"}
    path_cl = {result.model: result.cl_mae for result in results if result.split == "path"}
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "model",
            "train_n",
            "val_n",
            "cp_mae",
            "cl_mae",
            "cd_mae",
            "cd_drag_counts",
            "cl_degradation_vs_path",
            "cd_degradation_vs_path",
            "neuralfoil_confidence_mean",
            "neuralfoil_confidence_min",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "split": result.split,
                    "model": result.model,
                    "train_n": result.train_n,
                    "val_n": result.val_n,
                    "cp_mae": "" if result.cp_mae is None else f"{result.cp_mae:.8f}",
                    "cl_mae": f"{result.cl_mae:.8f}",
                    "cd_mae": f"{result.cd_mae:.8f}",
                    "cd_drag_counts": f"{result.cd_mae / 1e-4:.4f}",
                    "cl_degradation_vs_path": _ratio(result.cl_mae, path_cl.get(result.model)),
                    "cd_degradation_vs_path": _ratio(result.cd_mae, path_cd.get(result.model)),
                    "neuralfoil_confidence_mean": _optional_float(result.neuralfoil_confidence_mean),
                    "neuralfoil_confidence_min": _optional_float(result.neuralfoil_confidence_min),
                }
            )


def _optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def _ratio(value: float, base: float | None) -> str:
    if base is None or base == 0:
        return ""
    return f"{value / base:.6f}"


def _write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, data_path: Path, results: list[EvalResult], skipped: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OOD Comparison Report",
        "",
        f"Dataset: `{data_path}`",
        "",
        "| Split | Model | Train n | Val n | Cp MAE | Cl MAE | Cd MAE | Cd counts | NF conf mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        cp = "" if result.cp_mae is None else f"{result.cp_mae:.6f}"
        conf = "" if result.neuralfoil_confidence_mean is None else f"{result.neuralfoil_confidence_mean:.4f}"
        lines.append(
            f"| {result.split} | {result.model} | {result.train_n} | {result.val_n} | "
            f"{cp} | {result.cl_mae:.6f} | {result.cd_mae:.6f} | {result.cd_mae / 1e-4:.2f} | {conf} |"
        )
    if skipped:
        lines.extend(["", "## Skipped Splits", ""])
        for split, reason in skipped:
            lines.append(f"- `{split}`: {reason}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_separation_summary(path: Path, prediction_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["split"], row["model"]), []).append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "model",
            "n",
            "mean_cp_recovery_gradient_after",
            "mean_max_positive_dcpdx_after",
            "corr_recovery_gradient_cd_error",
            "corr_recovery_gradient_cl_error",
            "corr_max_dcpdx_cd_error",
            "corr_max_dcpdx_cl_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (split, model), rows in grouped.items():
            recovery = np.array([row["cp_recovery_gradient_after"] for row in rows], dtype=np.float32)
            max_dcpdx = np.array([row["max_positive_dcpdx_after"] for row in rows], dtype=np.float32)
            cd_error = np.array([row["cd_abs_error"] for row in rows], dtype=np.float32)
            cl_error = np.array([row["cl_abs_error"] for row in rows], dtype=np.float32)
            writer.writerow(
                {
                    "split": split,
                    "model": model,
                    "n": len(rows),
                    "mean_cp_recovery_gradient_after": f"{float(np.mean(recovery)):.8f}",
                    "mean_max_positive_dcpdx_after": f"{float(np.mean(max_dcpdx)):.8f}",
                    "corr_recovery_gradient_cd_error": f"{pearson_corr(recovery, cd_error):.8f}",
                    "corr_recovery_gradient_cl_error": f"{pearson_corr(recovery, cl_error):.8f}",
                    "corr_max_dcpdx_cd_error": f"{pearson_corr(max_dcpdx, cd_error):.8f}",
                    "corr_max_dcpdx_cl_error": f"{pearson_corr(max_dcpdx, cl_error):.8f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("aeromorph_flow/data/processed/xfoil_10k_transitions.npz"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--out-dir", type=Path, default=Path("aeromorph_flow/reports/ood_xfoil_10k"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--neuralfoil-model-size", type=str, default="xlarge")
    parser.add_argument("--neuralfoil-n-crit", type=float, default=9.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arrays = load_npz(args.data)
    results: list[EvalResult] = []
    prediction_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for split in args.splits:
        try:
            train_arrays, val_arrays = _make_split(arrays, split)
        except Exception as exc:
            skipped.append((split, str(exc)))
            print(f"skip split={split} reason={exc}")
            continue
        train_n = len(train_arrays["aoa"])
        val_n = len(val_arrays["aoa"])
        if train_n == 0 or val_n == 0:
            skipped.append((split, f"empty split train_n={train_n} val_n={val_n}"))
            print(f"skip split={split} train_n={train_n} val_n={val_n}")
            continue
        print(f"split={split} train_n={train_n} val_n={val_n}")
        for evaluator in [
            lambda: _eval_baseline(split, train_arrays, val_arrays, args.epochs, args.batch_size, args.hidden_dim),
            lambda: _eval_delta(split, train_arrays, val_arrays, args.epochs, args.batch_size, args.hidden_dim, 1.0, "delta"),
            lambda: _eval_delta(
                split,
                train_arrays,
                val_arrays,
                args.epochs,
                args.batch_size,
                args.hidden_dim,
                1000.0,
                "delta_cd_weighted",
            ),
            lambda: _eval_neuralfoil(split, train_n, val_arrays, args.neuralfoil_model_size, args.neuralfoil_n_crit),
        ]:
            result, rows = evaluator()
            results.append(result)
            prediction_rows.extend(rows)
            print(
                f"{split} {result.model} cl_mae={result.cl_mae:.6f} "
                f"cd_mae={result.cd_mae:.6f}"
            )

    _write_metrics(args.out_dir / "ood_metrics.csv", results)
    _write_predictions(args.out_dir / "ood_predictions.csv", prediction_rows)
    _write_separation_summary(args.out_dir / "separation_proxy_summary.csv", prediction_rows)
    _write_report(args.out_dir / "ood_report.md", args.data, results, skipped)
    print(f"wrote_metrics={args.out_dir / 'ood_metrics.csv'}")
    print(f"wrote_predictions={args.out_dir / 'ood_predictions.csv'}")
    print(f"wrote_separation_summary={args.out_dir / 'separation_proxy_summary.csv'}")
    print(f"wrote_report={args.out_dir / 'ood_report.md'}")


if __name__ == "__main__":
    main()
