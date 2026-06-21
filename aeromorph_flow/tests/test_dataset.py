from aeromorph_flow.src.training.dataset import DatasetConfig, DeltaDataset, generate_transition_dataset


def test_dataset_generation():
    arrays = generate_transition_dataset(DatasetConfig(n_paths=2, n_steps=3, n_points=32))
    ds = DeltaDataset(arrays)
    x, y = ds[0]
    assert len(ds) == 4
    assert x.ndim == 1
    assert y.ndim == 1
