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

For larger datasets, use the resumable batched generator. It writes chunk files first, skips
existing chunks on rerun, and then merges converged transitions into one `.npz` file:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.generate_xfoil_batched --target-transitions 10000 --batch-paths 250 --n-steps 5 --n-points 96 --seed 41 --xfoil-path 'E:\Code\Aerophflow\tools\Xfoil\bin\xfoil.exe' --xfoil-timeout-s 60 --xfoil-n-iter 120 --out aeromorph_flow/data/processed/xfoil_10k_transitions.npz --work-dir aeromorph_flow/data/processed/xfoil_10k_chunks --failure-log-dir aeromorph_flow/reports/xfoil_10k_failures
```

## Drag-Focused Training

Both training scripts support scalar loss weights for the final `Cl` and `Cd` outputs:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.train_delta --data aeromorph_flow/data/processed/xfoil_small_2000_transitions.npz --epochs 10 --batch-size 64 --hidden-dim 128 --seed 7 --cd-loss-weight 1000
```

On the 1864-sample XFOIL path split, increasing the Cd loss weight improved drag accuracy
while trading off some Cp/Cl accuracy:

```text
delta model, 10 epochs, unweighted:
cp_mae=0.023622, cl_mae=0.010968, cd_mae=0.000610

delta model, 10 epochs, --cd-loss-weight 1000:
cp_mae=0.031068, cl_mae=0.020701, cd_mae=0.000377
```

The delta trainer also supports a pressure-lift consistency regularizer:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.train_delta --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --epochs 30 --batch-size 128 --hidden-dim 128 --seed 7 --cp-cl-consistency-weight 1.0
```

This term encourages the scalar `delta_Cl` prediction to agree with an approximate chordwise
integral of predicted `delta_Cp_lower - delta_Cp_upper`.

## NeuralFoil Comparison

Evaluate NeuralFoil against the trained baseline and delta checkpoints on the same path-level
XFOIL validation split:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.evaluation.neuralfoil_baseline --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --out aeromorph_flow/reports/neuralfoil_10k_comparison.csv
```

Current 10k XFOIL path-split comparison:

```text
NeuralFoil:
cl_mae=0.005542, cd_mae=0.000167, cd_drag_counts=1.67

AeroMorph baseline:
cl_mae=0.009431, cd_mae=0.002403, cd_drag_counts=24.03

AeroMorph delta, unweighted:
cl_mae=0.007782, cd_mae=0.000393, cd_drag_counts=3.93

AeroMorph delta, --cd-loss-weight 1000 final checkpoint:
cl_mae=0.008291, cd_mae=0.000224, cd_drag_counts=2.24

AeroMorph delta, --cp-cl-consistency-weight 1.0:
cl_mae=0.008339, cd_mae=0.000359, cd_drag_counts=3.59

AeroMorph delta, --cp-cl-consistency-weight 1.0 --cd-loss-weight 1000 final checkpoint:
cl_mae=0.008232, cd_mae=0.000224, cd_drag_counts=2.24
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
