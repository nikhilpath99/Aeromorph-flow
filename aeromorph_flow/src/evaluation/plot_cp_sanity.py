from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aeromorph_flow.src.utils.io import load_npz


def _plot_cp_pair(path: Path, x: np.ndarray, cp_before: np.ndarray, cp_after: np.ndarray, title: str) -> None:
    n = len(x)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cp, label in [(axes[0], cp_before, "before"), (axes[1], cp_after, "after")]:
        ax.plot(x, cp[:n], label="upper")
        ax.plot(x, cp[n:], label="lower")
        ax.invert_yaxis()
        ax.set_title(label)
        ax.set_xlabel("x/c")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cp")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("aeromorph_flow/data/processed/xfoil_2000_transitions.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("aeromorph_flow/reports/cp_sanity"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    arrays = load_npz(args.data)
    n_samples = len(arrays["cp_before"])
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(n_samples, size=min(args.count, n_samples), replace=False)
    cp_dim = arrays["cp_before"].shape[1]
    x = np.linspace(0.0, 1.0, cp_dim // 2, dtype=np.float32)

    for i, sample_index in enumerate(chosen):
        title = (
            f"sample={sample_index} path={int(arrays['path_id'][sample_index, 0])} "
            f"step={int(arrays['step_index'][sample_index, 0])} "
            f"aoa={float(arrays['aoa'][sample_index, 0]):.2f} "
            f"Re={float(arrays['reynolds'][sample_index, 0]):.0f}"
        )
        _plot_cp_pair(
            args.out_dir / f"cp_sanity_{i:02d}.png",
            x,
            arrays["cp_before"][sample_index],
            arrays["cp_after"][sample_index],
            title,
        )

    print(f"wrote_cp_plots={args.out_dir}")


if __name__ == "__main__":
    main()
