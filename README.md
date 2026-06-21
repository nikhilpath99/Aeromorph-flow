# AeroMorph-Flow

Retrieval-augmented physics-memory experiments for morphing airfoil flow prediction.

This MVP runs without XFOIL by using a deterministic mock aerodynamic solver. It creates
NACA 4-digit morphing transitions, trains a small delta model, and reports validation
errors for `delta_cp`, `delta_cl`, and `delta_cd`.

## Quick Start

Use the existing global virtual environment:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.train_delta --epochs 5 --samples 128
```

## XFOIL Training

The local XFOIL 6.97 source archive has been built into:

```text
tools/Xfoil/bin/xfoil.exe
```

Run the XFOIL-backed delta model training with:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.train_delta --solver xfoil --xfoil-path 'E:\Code\Aerophflow\tools\Xfoil\bin\xfoil.exe' --xfoil-timeout-s 60 --epochs 5 --samples 32 --batch-size 8 --save-data aeromorph_flow/data/processed/xfoil_transitions.npz
```

## Baseline vs Delta Comparison

Generate a shared XFOIL transition dataset:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.train_delta --solver xfoil --xfoil-path 'E:\Code\Aerophflow\tools\Xfoil\bin\xfoil.exe' --xfoil-timeout-s 60 --epochs 1 --samples 64 --batch-size 16 --save-data aeromorph_flow/data/processed/xfoil_compare_transitions.npz
```

Train the absolute baseline on that same dataset:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.train_baseline --data aeromorph_flow/data/processed/xfoil_compare_transitions.npz --epochs 20 --batch-size 16 --hidden-dim 128 --seed 7
```

Train the delta model on that same dataset:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.train_delta --data aeromorph_flow/data/processed/xfoil_compare_transitions.npz --epochs 20 --batch-size 16 --hidden-dim 128 --seed 7
```

Useful XFOIL quality-control options:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.train_delta --solver xfoil --xfoil-path 'E:\Code\Aerophflow\tools\Xfoil\bin\xfoil.exe' --xfoil-n-iter 120 --xfoil-timeout-s 60 --samples 64 --epochs 5 --failure-log aeromorph_flow/reports/xfoil_failures.jsonl
```

Training now saves both the best-validation checkpoint and the final checkpoint:

```text
aeromorph_flow/data/processed/delta_mlp.pt
aeromorph_flow/data/processed/delta_mlp_last.pt
```

Create Cp sanity plots from a generated dataset:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.evaluation.plot_cp_sanity --data aeromorph_flow/data/processed/xfoil_2000_transitions.npz --out-dir aeromorph_flow/reports/cp_sanity --count 8
```

Current scaled XFOIL path-split result:

```text
dataset: aeromorph_flow/data/processed/xfoil_scaled_transitions.npz
samples: 473 converged transitions across 127 morph paths
split: path-level, 375 train transitions / 98 validation transitions

absolute baseline, 30 epochs:
val_mse=0.002869, cp_mae=0.035565, cl_mae=0.027225, cd_mae=0.004684

delta model, 30 epochs:
val_mse=0.000814, cp_mae=0.016842, cl_mae=0.012341, cd_mae=0.002946
```

Current 2000-request XFOIL path-split result:

```text
dataset: aeromorph_flow/data/processed/xfoil_2000_transitions.npz
samples: 1839 converged transitions across 498 morph paths
split: path-level, 1479 train transitions / 360 validation transitions

absolute baseline, 30 epochs, final:
val_mse=0.001910, cp_mae=0.028619, cl_mae=0.022637, cd_mae=0.004718

delta model, 30 epochs, final:
val_mse=0.000772, cp_mae=0.016129, cl_mae=0.013316, cd_mae=0.001992

delta model, best observed validation epoch:
epoch=29, val_mse=0.000433, cp_mae=0.011166, cl_mae=0.008644, cd_mae=0.001314
```

High-thickness extrapolation result:

```text
dataset: aeromorph_flow/data/processed/xfoil_2000_transitions.npz
split: train on transitions with max(thickness_before, thickness_after) <= 0.14
       validate on transitions with min(thickness_before, thickness_after) >= 0.15
gap: no samples with mixed/near-boundary thickness are used in either split
train_n=1045, val_n=355

absolute baseline, 30 epochs:
val_mse=0.005534, cp_mae=0.050892, cl_mae=0.035223, cd_mae=0.018564

delta model, 30 epochs:
val_mse=0.000761, cp_mae=0.016241, cl_mae=0.013311, cd_mae=0.001668
```

## Project Layout

```text
aeromorph_flow/
    data/
        raw/
        processed/
    notebooks/
    src/
        geometry/
        solvers/
        memory/
        models/
        training/
        evaluation/
        utils/
    tests/
```
