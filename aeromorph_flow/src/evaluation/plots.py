from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_cp(x: np.ndarray, cp: np.ndarray, label: str = "Cp"):
    fig, ax = plt.subplots()
    ax.plot(x, cp, label=label)
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.legend()
    return fig

