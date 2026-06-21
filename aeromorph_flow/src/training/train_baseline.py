from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aeromorph_flow.src.models.baseline_mlp import MLP
from aeromorph_flow.src.training.losses import prediction_loss_with_cd_penalty
from aeromorph_flow.src.training.dataset import (
    BaselineDataset,
    DatasetConfig,
    generate_transition_dataset,
    split_arrays,
    split_arrays_extrapolation,
    split_arrays_by_path,
)
from aeromorph_flow.src.utils.io import load_npz, save_npz


def evaluate(model: torch.nn.Module, loader: DataLoader, cp_dim: int) -> dict[str, float]:
    model.eval()
    losses = []
    cp_mae = []
    cl_mae = []
    cd_mae = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            err = pred - y
            losses.append(torch.mean(err**2).item())
            cp_mae.append(torch.mean(torch.abs(err[:, :cp_dim])).item())
            cl_mae.append(torch.mean(torch.abs(err[:, cp_dim])).item())
            cd_mae.append(torch.mean(torch.abs(err[:, cp_dim + 1])).item())
    return {
        "mse": float(np.mean(losses)),
        "cp_mae": float(np.mean(cp_mae)),
        "cl_mae": float(np.mean(cl_mae)),
        "cd_mae": float(np.mean(cd_mae)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=128, help="Approximate transition count.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--solver", choices=["mock", "xfoil"], default="mock")
    parser.add_argument("--xfoil-path", type=str, default=None)
    parser.add_argument("--xfoil-timeout-s", type=float, default=30.0)
    parser.add_argument("--data", type=Path, default=None, help="Use an existing .npz dataset instead of generating.")
    parser.add_argument("--save-data", type=Path, default=None)
    parser.add_argument(
        "--split-by",
        choices=["path", "sample", "ood_thickness_high", "ood_camber_high", "ood_morph_large"],
        default="path",
    )
    parser.add_argument("--cd-penalty-weight", type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.data is not None:
        arrays = load_npz(args.data)
        save_data = args.data
    else:
        n_steps = 5
        n_paths = max(2, int(np.ceil(args.samples / (n_steps - 1))))
        config = DatasetConfig(
            n_paths=n_paths,
            n_steps=n_steps,
            seed=args.seed,
            solver=args.solver,
            xfoil_path=args.xfoil_path,
            xfoil_timeout_s=args.xfoil_timeout_s,
        )
        save_data = args.save_data
        if save_data is None:
            save_data = Path(f"aeromorph_flow/data/processed/{args.solver}_transitions.npz")
        arrays = generate_transition_dataset(config)
        save_npz(save_data, arrays)

    if args.split_by == "path":
        train_arrays, val_arrays = split_arrays_by_path(arrays, val_fraction=0.2, seed=args.seed + 1)
    elif args.split_by == "sample":
        train_arrays, val_arrays = split_arrays(arrays, val_fraction=0.2, seed=args.seed + 1)
    else:
        train_arrays, val_arrays = split_arrays_extrapolation(arrays, mode=args.split_by)
    print(f"split_by={args.split_by} train_n={len(next(iter(train_arrays.values())))} val_n={len(next(iter(val_arrays.values())))}")
    train_ds = BaselineDataset(train_arrays)
    val_ds = BaselineDataset(val_arrays)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    input_dim = train_ds.x.shape[1]
    output_dim = train_ds.y.shape[1]
    cp_dim = arrays["cp_after"].shape[1]
    cd_index = cp_dim + 1
    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            pred = model(x)
            loss = prediction_loss_with_cd_penalty(
                pred,
                y,
                cd_index=cd_index,
                cd_penalty_weight=args.cd_penalty_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        metrics = evaluate(model, val_loader, cp_dim=cp_dim)
        print(
            f"epoch={epoch:03d} train_mse={np.mean(train_losses):.6f} "
            f"val_mse={metrics['mse']:.6f} cp_mae={metrics['cp_mae']:.6f} "
            f"cl_mae={metrics['cl_mae']:.6f} cd_mae={metrics['cd_mae']:.6f}"
        )

    model_path = Path("aeromorph_flow/data/processed/baseline_mlp.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "output_dim": output_dim}, model_path)
    print(f"saved_data={save_data}")
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    main()
