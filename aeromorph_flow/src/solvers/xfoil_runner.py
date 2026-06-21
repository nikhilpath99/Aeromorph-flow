from __future__ import annotations

import shutil
import subprocess
import tempfile
import os
from pathlib import Path

import numpy as np


def find_xfoil_executable(xfoil_path: str | Path | None = None) -> str:
    """Resolve an XFOIL executable from an explicit path or PATH."""
    if xfoil_path is not None:
        path = Path(xfoil_path)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"XFOIL executable was not found at {path}.")

    resolved = shutil.which("xfoil") or shutil.which("xfoil.exe")
    if resolved:
        return resolved
    raise FileNotFoundError("XFOIL executable was not found on PATH. Pass --xfoil-path if it is elsewhere.")


def _write_airfoil_dat(path: Path, airfoil: dict) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"{airfoil.get('code', 'airfoil')}\n")
        for x, y in np.asarray(airfoil["coords"], dtype=float):
            handle.write(f"{x:.8f} {y:.8f}\n")


def _parse_polar(path: Path) -> dict[str, float]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append([float(value) for value in parts[:5]])
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"XFOIL did not produce a readable polar file at {path}.")
    alpha, cl, cd, _cdp, cm = rows[-1]
    return {"alpha": alpha, "cl": cl, "cd": cd, "cm": cm}


def _interp_surface(x: np.ndarray, cp: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = np.asarray(x[order], dtype=float)
    cp_sorted = np.asarray(cp[order], dtype=float)
    unique_x, inverse = np.unique(np.round(x_sorted, 8), return_inverse=True)
    cp_unique = np.zeros_like(unique_x, dtype=float)
    counts = np.zeros_like(unique_x, dtype=float)
    for source_index, target_index in enumerate(inverse):
        cp_unique[target_index] += cp_sorted[source_index]
        counts[target_index] += 1.0
    cp_unique /= np.maximum(counts, 1.0)
    return np.interp(target_x, unique_x, cp_unique, left=cp_unique[0], right=cp_unique[-1])


def _parse_cp(path: Path, target_x: np.ndarray) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values = [float(value) for value in parts]
        except ValueError:
            continue
        # XFOIL CPWR commonly emits x/c Cp; some builds include y/c as x y Cp.
        x = values[0]
        cp = values[-1]
        if -0.25 <= x <= 1.25 and -20.0 <= cp <= 20.0:
            rows.append((x, cp))

    if len(rows) < 8:
        raise RuntimeError(f"XFOIL did not produce enough Cp samples at {path}.")

    arr = np.asarray(rows, dtype=float)
    x = arr[:, 0]
    cp = arr[:, 1]
    le_index = int(np.argmin(x))
    upper_x = x[: le_index + 1]
    upper_cp = cp[: le_index + 1]
    lower_x = x[le_index:]
    lower_cp = cp[le_index:]
    if len(upper_x) < 4 or len(lower_x) < 4:
        midpoint = len(x) // 2
        upper_x, upper_cp = x[:midpoint], cp[:midpoint]
        lower_x, lower_cp = x[midpoint:], cp[midpoint:]

    cp_upper = _interp_surface(upper_x, upper_cp, target_x)
    cp_lower = _interp_surface(lower_x, lower_cp, target_x)
    return np.concatenate([cp_upper, cp_lower]).astype(np.float32)


def solve_airfoil_xfoil(
    airfoil: dict,
    aoa_deg: float,
    reynolds: float,
    xfoil_path: str | Path | None = None,
    n_iter: int = 100,
    timeout_s: float = 30.0,
) -> dict:
    """Run XFOIL for one airfoil condition and return fixed-grid Cp/Cl/Cd/Cm."""
    executable = find_xfoil_executable(xfoil_path)

    with tempfile.TemporaryDirectory(prefix="aeromorph_xfoil_") as tmp:
        tmp_dir = Path(tmp)
        airfoil_path = tmp_dir / "airfoil.dat"
        polar_path = tmp_dir / "polar.txt"
        cp_path = tmp_dir / "cp.txt"
        _write_airfoil_dat(airfoil_path, airfoil)

        commands = "\n".join(
            [
                f"LOAD {airfoil_path.name}",
                "PANE",
                "OPER",
                f"VISC {float(reynolds):.6g}",
                f"ITER {int(n_iter)}",
                "PACC",
                polar_path.name,
                "",
                f"ALFA {float(aoa_deg):.6g}",
                f"CPWR {cp_path.name}",
                "",
                "QUIT",
                "",
            ]
        )
        env = os.environ.copy()
        msys_mingw_bin = Path("C:/msys64/mingw64/bin")
        if msys_mingw_bin.exists():
            env["PATH"] = f"{msys_mingw_bin}{os.pathsep}{env.get('PATH', '')}"

        proc = subprocess.run(
            [executable],
            input=commands,
            text=True,
            capture_output=True,
            cwd=tmp_dir,
            env=env,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"XFOIL failed with code {proc.returncode}: {proc.stderr or proc.stdout}")
        if not polar_path.exists() or not cp_path.exists():
            raise RuntimeError(f"XFOIL did not converge/write outputs. Last output:\n{proc.stdout[-2000:]}")

        polar = _parse_polar(polar_path)
        cp = _parse_cp(cp_path, np.asarray(airfoil["x"], dtype=float))
        return {
            "cp": cp,
            "cl": float(polar["cl"]),
            "cd": float(polar["cd"]),
            "cm": float(polar["cm"]),
            "converged": True,
        }
