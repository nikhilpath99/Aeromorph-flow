from __future__ import annotations

import numpy as np


class MemoryBank:
    def __init__(self) -> None:
        self.keys: list[np.ndarray] = []
        self.values: list[dict] = []

    def add(self, key: np.ndarray, value: dict) -> None:
        self.keys.append(np.asarray(key, dtype=np.float32))
        self.values.append(value)

    def as_arrays(self) -> tuple[np.ndarray, list[dict]]:
        if not self.keys:
            return np.empty((0, 0), dtype=np.float32), []
        return np.stack(self.keys).astype(np.float32), self.values

