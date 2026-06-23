# Experiment Notes

This file keeps detailed results and commands out of the main README.

## XFOIL Dataset

Generated with the resumable batched XFOIL generator:

```bash
python -m aeromorph_flow.src.training.generate_xfoil_batched --target-transitions 10000 --batch-paths 250 --n-steps 5 --n-points 96 --seed 41 --xfoil-path /path/to/xfoil --xfoil-timeout-s 60 --xfoil-n-iter 120 --out aeromorph_flow/data/processed/xfoil_10k_transitions.npz --work-dir aeromorph_flow/data/processed/xfoil_10k_chunks --failure-log-dir aeromorph_flow/reports/xfoil_10k_failures
```

Summary:

```text
target transitions: 10000
requested morph paths: 2500
converged transitions: 9334
unique paths: 2494
XFOIL failure count: 453
AoA range: -1.999 to 7.989 deg
log10(Re) range: 5.699 to 6.301
```

## Path-Split Coefficient Results

XFOIL 10k validation split:

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

AeroMorph delta, --path-consistency-weight 0.1 --cd-loss-weight 1000:
cl_mae=0.008089, cd_mae=0.000279, cd_drag_counts=2.79
```

## Path-Consistency Endpoint Reconstruction

Held-out morph paths: 490.

```text
delta unweighted:
endpoint_cp_mae=0.024147, endpoint_cl_error=0.019794, endpoint_cd_error=0.001086

delta, --cp-cl-consistency-weight 1.0:
endpoint_cp_mae=0.023691, endpoint_cl_error=0.023743, endpoint_cd_error=0.000993

delta, --path-consistency-weight 0.1:
endpoint_cp_mae=0.022975, endpoint_cl_error=0.019230, endpoint_cd_error=0.002320

delta, --path-consistency-weight 0.1 --cd-loss-weight 1000:
endpoint_cp_mae=0.031702, endpoint_cl_error=0.021330, endpoint_cd_error=0.000791
```

## OOD Comparison

Generated with:

```bash
python -m aeromorph_flow.src.evaluation.ood_comparison --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --epochs 30 --batch-size 128 --splits path ood_thickness_high ood_camber_high ood_morph_large ood_aoa_high ood_re_high --out-dir aeromorph_flow/reports/ood_xfoil_10k
```

Summary:

```text
path:
baseline cd_mae=0.004491
delta cd_mae=0.000339
delta_cd_weighted cd_mae=0.000351
neuralfoil cd_mae=0.000167

ood_thickness_high:
baseline cd_mae=0.003756
delta cd_mae=0.000602
delta_cd_weighted cd_mae=0.000271
neuralfoil cd_mae=0.000180

ood_camber_high:
baseline cd_mae=0.003616
delta cd_mae=0.000994
delta_cd_weighted cd_mae=0.000585
neuralfoil cd_mae=0.000273

ood_morph_large:
baseline cd_mae=0.000943
delta cd_mae=0.000816
delta_cd_weighted cd_mae=0.000339
neuralfoil cd_mae=0.000176

ood_aoa_high:
baseline cd_mae=0.012048
delta cd_mae=0.002110
delta_cd_weighted cd_mae=0.001058
neuralfoil cd_mae=0.000343

ood_re_high:
baseline cd_mae=0.003235
delta cd_mae=0.000772
delta_cd_weighted cd_mae=0.000493
neuralfoil cd_mae=0.000159
```

Interpretation:

- Delta models strongly improve drag prediction over the absolute baseline on every OOD split.
- Cd weighting is the most robust AeroMorph variant under high thickness, high camber, large morph, high AoA, and high Re.
- High AoA is the hardest split for AeroMorph drag prediction.
- NeuralFoil remains the strongest coefficient predictor across these one-step OOD tests.

## Useful Commands

Train absolute baseline:

```bash
python -m aeromorph_flow.src.training.train_baseline --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --epochs 30 --batch-size 128 --hidden-dim 128 --seed 7
```

Train unweighted delta:

```bash
python -m aeromorph_flow.src.training.train_delta --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --epochs 30 --batch-size 128 --hidden-dim 128 --seed 7
```

Train Cd-weighted delta:

```bash
python -m aeromorph_flow.src.training.train_delta --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --epochs 30 --batch-size 128 --hidden-dim 128 --seed 7 --cd-loss-weight 1000
```

Train pressure-lift consistency delta:

```bash
python -m aeromorph_flow.src.training.train_delta --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --epochs 30 --batch-size 128 --hidden-dim 128 --seed 7 --cp-cl-consistency-weight 1.0
```

Train path-consistency delta:

```bash
python -m aeromorph_flow.src.training.train_delta --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --epochs 30 --batch-size 128 --hidden-dim 128 --seed 7 --path-consistency-weight 0.1 --path-consistency-max-paths 256
```

Evaluate NeuralFoil:

```bash
python -m aeromorph_flow.src.evaluation.neuralfoil_baseline --data aeromorph_flow/data/processed/xfoil_10k_transitions.npz --out aeromorph_flow/reports/neuralfoil_10k_comparison.csv
```
