from __future__ import annotations

import numpy as np


def retrieve_top_k(query: np.ndarray, keys: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy nearest-neighbor retrieval."""
    if keys.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    distances = np.linalg.norm(keys - query[None, :], axis=1)
    idx = np.argsort(distances)[:k]
    return idx.astype(np.int64), distances[idx].astype(np.float32)

