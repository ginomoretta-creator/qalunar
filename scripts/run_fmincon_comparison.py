"""Cross-validate the Python continuous baseline against MATLAB fmincon.

Both solvers tackle the *same* energy-optimal planar CR3BP transfer with the
*same* Hermite-Simpson direct transcription (objective J = 1/2 integral ||u||^2 dt):

* Python: ``qalunar.reference.direct_collocation.solve_energy_optimal_cr3bp``
  (scipy SLSQP).
* MATLAB: ``matlab/cr3bp_fmincon.m`` (fmincon SQP).

The script runs the Python solve, shells out to MATLAB via ``matlab -batch``,
reads the fmincon result back, and prints a side-by-side comparison of the
objective, constraint satisfaction, and node-wise trajectory difference. Close
agreement is an independent, cross-language confirmation that the continuous
baseline is correct.

Requires MATLAB on PATH with the Optimization Toolbox. Run::

    python -m scripts.run_fmincon_comparison
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.reference.direct_collocation import (
    DirectCollocationConfig,
    solve_energy_optimal_cr3bp,
)


# Known-converging case (mild BCs away from the primaries), mirroring the
# fixture in tests/test_qubo_sampling.py.
CASE = dict(
    r0=np.array([-0.3, 0.0]),
    v0=np.array([0.0, 0.6]),
    rf=np.array([0.4, 0.2]),
    vf=np.array([-0.1, 0.0]),
    time_of_flight=2.5,
)
N_INTERVALS = 40
MAXITER = 300
TOL = 1e-8

REPO = Path(__file__).resolve().parents[1]
MATLAB_DIR = REPO / "matlab"
IO_DIR = MATLAB_DIR / "_io"


def _run_matlab(params_path: Path, results_path: Path) -> None:
    matlab = shutil.which("matlab")
    if matlab is None:
        sys.exit("MATLAB not found on PATH; cannot run the fmincon comparison.")
    cmd = (
        f"cr3bp_fmincon('{params_path.as_posix()}', "
        f"'{results_path.as_posix()}')"
    )
    print(f"  invoking MATLAB fmincon ({cmd}) ...")
    proc = subprocess.run(
        [matlab, "-batch", cmd],
        cwd=str(MATLAB_DIR),
        capture_output=True,
        text=True,
    )
    if proc.stdout.strip():
        print("  [matlab stdout]", proc.stdout.strip())
    if proc.returncode != 0:
        print("  [matlab stderr]", proc.stderr.strip())
        sys.exit(f"MATLAB exited with code {proc.returncode}")


def main() -> None:
    dyn = PlanarCR3BP(mu=EARTH_MOON_MU)
    cfg = DirectCollocationConfig(n_intervals=N_INTERVALS, maxiter=MAXITER, tol=TOL)

    print("=" * 70)
    print("  Continuous baseline: Python SLSQP  vs  MATLAB fmincon")
    print("=" * 70)
    print(f"  case: r0={CASE['r0']}, v0={CASE['v0']}, "
          f"rf={CASE['rf']}, vf={CASE['vf']}, T={CASE['time_of_flight']}")
    print(f"  transcription: Hermite-Simpson, {N_INTERVALS} intervals\n")

    # --- Python SLSQP ---
    py = solve_energy_optimal_cr3bp(
        dyn, CASE["r0"], CASE["v0"], CASE["rf"], CASE["vf"],
        CASE["time_of_flight"], config=cfg,
    )

    # --- MATLAB fmincon ---
    IO_DIR.mkdir(parents=True, exist_ok=True)
    params_path = IO_DIR / "params.json"
    results_path = IO_DIR / "results.json"
    params = dict(
        mu=EARTH_MOON_MU,
        r0=CASE["r0"].tolist(), v0=CASE["v0"].tolist(),
        rf=CASE["rf"].tolist(), vf=CASE["vf"].tolist(),
        T=CASE["time_of_flight"],
        n_intervals=N_INTERVALS, maxiter=MAXITER, tol=TOL,
        control_bound=None, algorithm="sqp",
    )
    params_path.write_text(json.dumps(params), encoding="utf-8")
    _run_matlab(params_path, results_path)
    ml = json.loads(results_path.read_text(encoding="utf-8"))

    # --- Compare ---
    def col(a, b):
        return f"{a:>18}{b:>18}"

    print()
    print("-" * 70)
    print(f"  {'':<22}{'Python SLSQP':>18}{'MATLAB fmincon':>18}")
    print("-" * 70)
    print(f"  {'objective J':<22}{py.objective:>18.9e}{ml['objective']:>18.9e}")
    print(f"  {'converged':<22}{str(py.success):>18}{str(bool(ml['success'])):>18}")
    print(f"  {'iterations':<22}{py.n_iterations:>18}{int(ml['n_iterations']):>18}")
    print(f"  {'max defect':<22}{py.max_defect:>18.3e}{ml['max_defect']:>18.3e}")
    print(f"  {'max BC error':<22}{py.max_bc_error:>18.3e}{ml['max_bc_error']:>18.3e}")

    # Node-wise trajectory difference (same uniform grid, same node count).
    px = np.column_stack([py.x, py.y, py.vx, py.vy, py.ux, py.uy])
    mx = np.column_stack([
        ml["x"], ml["y"], ml["vx"], ml["vy"], ml["ux"], ml["uy"],
    ])
    labels = ["x", "y", "vx", "vy", "ux", "uy"]

    print()
    print("-" * 70)
    print("  Node-wise RMS difference (Python - MATLAB)")
    print("-" * 70)
    for j, lab in enumerate(labels):
        rms = float(np.sqrt(np.mean((px[:, j] - mx[:, j]) ** 2)))
        print(f"  {lab:<6}{rms:.3e}")

    rel_obj = abs(py.objective - ml["objective"]) / max(abs(py.objective), 1e-30)
    # Feasibility: a solution is trustworthy only if its defects/BCs are tiny.
    py_feasible = py.max_defect < 1e-6 and py.max_bc_error < 1e-6
    ml_feasible = ml["max_defect"] < 1e-6 and ml["max_bc_error"] < 1e-6

    print()
    print(f"  relative objective difference: {rel_obj:.3e}")
    if rel_obj < 1e-4 and np.max(np.abs(px - mx)) < 1e-3:
        print("  => MATCH: the two optimizers agree; baseline cross-validated.")
    elif py_feasible and ml_feasible:
        better = "MATLAB fmincon" if ml["objective"] < py.objective else "Python SLSQP"
        print("  => DIFFERENT LOCAL MINIMA: both solutions are feasible "
              "(defects << 1e-6) but the")
        print(f"     objectives differ. {better} found the lower-cost optimum. "
              "The transcribed")
        print("     CR3BP NLP is nonconvex, so the 'classical optimum' is "
              "optimizer/seed-dependent.")
    else:
        print("  => DISCREPANCY with an INFEASIBLE solution: investigate "
              "(tolerances or a bug).")


if __name__ == "__main__":
    main()
