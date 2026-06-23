import numpy as np

from aeromorph_flow.src.training.dataset import (
    DatasetConfig,
    DeltaDataset,
    generate_transition_dataset,
    split_arrays_extrapolation,
)


def test_dataset_generation():
    arrays = generate_transition_dataset(DatasetConfig(n_paths=2, n_steps=3, n_points=32))
    ds = DeltaDataset(arrays)
    x, y = ds[0]
    assert len(ds) == 4
    assert x.ndim == 1
    assert y.ndim == 1


def test_ood_splits_are_nonempty():
    n = 12
    arrays = {
        "params_before": np.column_stack(
            [
                np.linspace(0.01, 0.06, n),
                np.full(n, 0.4),
                np.linspace(0.10, 0.18, n),
            ]
        ).astype(np.float32),
        "params_after": np.column_stack(
            [
                np.linspace(0.012, 0.062, n),
                np.full(n, 0.42),
                np.linspace(0.105, 0.185, n),
            ]
        ).astype(np.float32),
        "delta_params": np.column_stack(
            [
                np.linspace(0.001, 0.050, n),
                np.zeros(n),
                np.zeros(n),
            ]
        ).astype(np.float32),
        "aoa": np.linspace(-1.0, 8.0, n, dtype=np.float32).reshape(-1, 1),
        "log_re": np.linspace(5.7, 6.3, n, dtype=np.float32).reshape(-1, 1),
    }
    for mode in [
        "ood_thickness_high",
        "ood_camber_high",
        "ood_morph_large",
        "ood_aoa_high",
        "ood_re_high",
    ]:
        train, val = split_arrays_extrapolation(arrays, mode)
        assert len(train["aoa"]) > 0
        assert len(val["aoa"]) > 0
