from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aeromorph_flow.src.training.dataset import DatasetConfig, generate_transition_dataset
from aeromorph_flow.src.utils.io import load_npz, save_npz


def _merge_chunks(chunk_paths: list[Path], out_path: Path) -> dict[str, np.ndarray]:
    chunks = [load_npz(path) for path in chunk_paths]
    sample_n = len(chunks[0]["path_id"])
    merged: dict[str, np.ndarray] = {}
    for key in chunks[0]:
        values = [chunk[key] for chunk in chunks]
        if len(values[0]) == sample_n:
            merged[key] = np.concatenate(values, axis=0)
        elif key == "failure_count":
            merged[key] = np.array([sum(int(value[0]) for value in values)], dtype=np.int64)
        elif key == "requested_paths":
            merged[key] = np.array([sum(int(value[0]) for value in values)], dtype=np.int64)
        else:
            merged[key] = values[0]
    save_npz(out_path, merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-transitions", type=int, default=101_000)
    parser.add_argument("--batch-paths", type=int, default=250)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--n-points", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--xfoil-path", type=str, required=True)
    parser.add_argument("--xfoil-timeout-s", type=float, default=60.0)
    parser.add_argument("--xfoil-n-iter", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("aeromorph_flow/data/processed/xfoil_101k_transitions.npz"))
    parser.add_argument("--work-dir", type=Path, default=Path("aeromorph_flow/data/processed/xfoil_101k_chunks"))
    parser.add_argument("--failure-log-dir", type=Path, default=Path("aeromorph_flow/reports/xfoil_101k_failures"))
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.failure_log_dir.mkdir(parents=True, exist_ok=True)

    paths_per_target = args.n_steps - 1
    requested_paths_total = int(np.ceil(args.target_transitions / paths_per_target))
    n_batches = int(np.ceil(requested_paths_total / args.batch_paths))
    chunk_paths: list[Path] = []
    converged_total = 0

    print(
        f"target_transitions={args.target_transitions} requested_paths={requested_paths_total} "
        f"batch_paths={args.batch_paths} batches={n_batches}"
    )

    for batch_index in range(n_batches):
        start_path = batch_index * args.batch_paths
        paths_this_batch = min(args.batch_paths, requested_paths_total - start_path)
        chunk_path = args.work_dir / f"chunk_{batch_index:04d}.npz"
        if chunk_path.exists():
            arrays = load_npz(chunk_path)
            converged = len(arrays["path_id"])
            print(f"batch={batch_index + 1}/{n_batches} status=skip existing transitions={converged}")
        else:
            failure_log = args.failure_log_dir / f"failures_{batch_index:04d}.jsonl"
            config = DatasetConfig(
                n_paths=paths_this_batch,
                n_steps=args.n_steps,
                n_points=args.n_points,
                seed=args.seed + batch_index,
                solver="xfoil",
                xfoil_path=args.xfoil_path,
                xfoil_timeout_s=args.xfoil_timeout_s,
                xfoil_n_iter=args.xfoil_n_iter,
                failure_log_path=str(failure_log),
            )
            arrays = generate_transition_dataset(config)
            arrays["global_path_offset"] = np.array([start_path], dtype=np.int64)
            arrays["path_id"] = arrays["path_id"] + start_path
            save_npz(chunk_path, arrays)
            converged = len(arrays["path_id"])
            print(
                f"batch={batch_index + 1}/{n_batches} status=done paths={paths_this_batch} "
                f"transitions={converged} failures={int(arrays['failure_count'][0])}"
            )
        chunk_paths.append(chunk_path)
        converged_total += converged

    merged = _merge_chunks(chunk_paths, args.out)
    print(f"saved={args.out}")
    print(f"converged_transitions={len(merged['path_id'])}")
    print(f"unique_paths={len(np.unique(merged['path_id']))}")
    print(f"failure_count={int(merged['failure_count'][0])}")


if __name__ == "__main__":
    main()
