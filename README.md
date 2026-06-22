# AeroMorph-Flow

AeroMorph-Flow is an early research prototype for learning **aerodynamic changes during
airfoil morphing**.

Most airfoil surrogate models learn an absolute mapping:

```text
airfoil + AoA + Re -> Cp, Cl, Cd
```

This project tests a local morphing-delta formulation:

```text
airfoil_before + morph_delta + flow_before -> delta_Cp, delta_Cl, delta_Cd
```

The goal is not to replace mature airfoil solvers globally. The goal is to test whether
conditioning on the previous aerodynamic state improves prediction of the next state during
smooth geometry changes.

## Current Status

- Built a reproducible XFOIL transition dataset with 9,334 converged samples across 2,494 morph paths.
- Compared an absolute MLP baseline, delta models, drag-weighted training, physics-inspired losses, and NeuralFoil.
- Added pressure-lift consistency and morph-path consistency regularizers.
- NeuralFoil remains stronger on standard one-step prediction, while AeroMorph-Flow strongly improves over the internal absolute baseline and gives a morphing-specific path-consistency testbed.

## Key 10k XFOIL Results

Path-level validation split, XFOIL as label source:

| Model | Cl MAE | Cd MAE | Cd error |
|---|---:|---:|---:|
| NeuralFoil | 0.005542 | 0.000167 | 1.67 drag counts |
| AeroMorph absolute baseline | 0.009431 | 0.002403 | 24.03 drag counts |
| AeroMorph delta | 0.007782 | 0.000393 | 3.93 drag counts |
| AeroMorph delta, Cd-weighted | 0.008291 | 0.000224 | 2.24 drag counts |

Path-consistency endpoint reconstruction on 490 held-out morph paths:

| Model | Endpoint Cp MAE | Endpoint Cl error | Endpoint Cd error |
|---|---:|---:|---:|
| Delta | 0.024147 | 0.019794 | 0.001086 |
| Delta + pressure-lift consistency | 0.023691 | 0.023743 | 0.000993 |
| Delta + path consistency | 0.022975 | 0.019230 | 0.002320 |
| Delta + path consistency + Cd weighting | 0.031702 | 0.021330 | 0.000791 |

See [docs/results.md](docs/results.md) for detailed commands and experiment notes.

## Why This Is Interesting

The strongest current evidence is that morphing-delta prediction is much better than a direct
absolute baseline for drag:

```text
absolute baseline Cd MAE: 0.002403
delta model Cd MAE:       0.000393
Cd-weighted delta MAE:    0.000224
NeuralFoil Cd MAE:        0.000167
```

The project is therefore best framed as:

```text
Can a transition-aware model improve local morphing predictions and multi-step morph consistency?
```

not as:

```text
Can a small prototype beat NeuralFoil globally?
```

## Quick Start

Install dependencies:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m pip install -r requirements.txt
```

Run a mock-data smoke test:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.train_delta --epochs 5 --samples 128
```

Generate a resumable XFOIL dataset:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.generate_xfoil_batched --target-transitions 10000 --batch-paths 250 --n-steps 5 --n-points 96 --seed 41 --xfoil-path 'E:\Code\Aerophflow\tools\Xfoil\bin\xfoil.exe' --xfoil-timeout-s 60 --xfoil-n-iter 120 --out aeromorph_flow/data/processed/xfoil_10k_transitions.npz --work-dir aeromorph_flow/data/processed/xfoil_10k_chunks --failure-log-dir aeromorph_flow/reports/xfoil_10k_failures
```

Train the delta model:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.training.train_delta --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --epochs 30 --batch-size 128 --hidden-dim 128 --seed 7
```

Evaluate NeuralFoil and AeroMorph checkpoints:

```powershell
& 'E:\Code\global_venv\Scripts\python.exe' -m aeromorph_flow.src.evaluation.neuralfoil_baseline --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --out aeromorph_flow/reports/neuralfoil_10k_comparison.csv
```

## Repository Layout

```text
aeromorph_flow/
  src/
    geometry/      NACA-like airfoil generation and morphing
    solvers/       mock solver and optional XFOIL wrapper
    training/      dataset generation and model training
    evaluation/    NeuralFoil, path consistency, OOD/report scripts
    models/        MLP baseline and delta models
    memory/        retrieval-memory prototype utilities
  tests/
docs/
```

## Limitations

- XFOIL is treated as the label source; it is not experimental or RANS truth.
- Current geometry space is NACA-like and low-dimensional.
- NeuralFoil is still better on ordinary one-step absolute prediction.
- Retrieval-memory modeling exists in the codebase but has not yet been developed into the main comparison.
- The physics regularizers are early and need better tuning.

## Next Steps

- Add stronger OOD evaluations for high camber, high thickness, and large morph directions.
- Improve path-consistency training without degrading Cd.
- Add retrieval over morphing transitions and compare against the best delta model.
- Validate selected cases with higher-fidelity CFD or experimental data.
