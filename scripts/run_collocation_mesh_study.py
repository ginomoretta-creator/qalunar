"""Mesh-refinement study of the direct-collocation "classical baseline".

The Hermite-Simpson NLP used as the classical reference for the QUBO
comparisons was reported at a single mesh (N = 40). A reference objective is
only meaningful if it is converged in the mesh *and* successive meshes stay in
the same basin of the (nonconvex) NLP. This script:

1. solves the benchmark case at N = 20, 40, 80, 160, **warm-starting each mesh
   from the previous solution** (linear interpolation of states and controls)
   so the sequence tracks one local minimum instead of hopping basins;
2. reports J(N), max defect, boundary error, the feasibility-gated success
   flag and SLSQP's raw flag;
3. estimates the converged value by Richardson extrapolation from the last
   two meshes (assuming the O(h^4) rate of Hermite-Simpson) and the
   discretisation error of the N = 40 value the papers quoted.

Case: the one shared by ``run_fmincon_comparison.py`` and
``run_hybrid_comparison.py``.

Run:  python -m scripts.run_collocation_mesh_study
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.reference.direct_collocation import (
    DirectCollocationConfig,
    _pack,
    solve_energy_optimal_cr3bp,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
R0, V0 = np.array([-0.3, 0.0]), np.array([0.0, 0.6])
RF, VF = np.array([0.4, 0.2]), np.array([-0.1, 0.0])
TOF = 2.5
MESHES = [20, 40, 80, 160]          # override with --meshes


def _warm_start(res, n_new: int) -> np.ndarray:
    t_old = res.t
    t_new = np.linspace(0.0, TOF, n_new + 1)
    cols = [np.interp(t_new, t_old, getattr(res, k)) for k in ("x", "y", "vx", "vy")]
    ctrl = [np.interp(t_new, t_old, getattr(res, k)) for k in ("ux", "uy")]
    return _pack(np.column_stack(cols), np.column_stack(ctrl))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--meshes', type=int, nargs='+', default=MESHES)
    args = ap.parse_args()
    meshes = list(args.meshes)
    dyn = PlanarCR3BP()
    rows, prev = [], None
    print(f"{'N':>5} {'J':>14} {'max defect':>11} {'bc err':>9} {'feasible':>9} {'slsqp':>6} {'iters':>6} {'time s':>7}")
    print("-" * 78)
    for N in meshes:
        cfg = DirectCollocationConfig(n_intervals=N, maxiter=1500, tol=1e-10, feasibility_tol=1e-7)
        guess = _warm_start(prev, N) if prev is not None else None
        t0 = time.perf_counter()
        res = solve_energy_optimal_cr3bp(dyn, R0, V0, RF, VF, TOF, config=cfg, initial_guess=guess)
        dt = time.perf_counter() - t0
        rows.append({"N": N, "J": res.objective, "max_defect": res.max_defect,
                     "max_bc_error": res.max_bc_error, "feasible": res.success,
                     "slsqp_success": res.slsqp_success, "iterations": res.n_iterations,
                     "time_s": dt, "warm_started": prev is not None})
        print(f"{N:>5} {res.objective:14.6e} {res.max_defect:11.2e} {res.max_bc_error:9.1e} "
              f"{str(res.success):>9} {str(res.slsqp_success):>6} {res.n_iterations:>6} {dt:7.1f}",
              flush=True)
        prev = res

    J = {r["N"]: r["J"] for r in rows}
    out = {"case": {"r0": R0.tolist(), "v0": V0.tolist(), "rf": RF.tolist(), "vf": VF.tolist(),
                    "tof": TOF}, "rows": rows}
    if all(r["feasible"] for r in rows[-2:]):
        J1, J2 = J[meshes[-2]], J[meshes[-1]]
        J_inf = J2 + (J2 - J1) / (2 ** 4 - 1)          # Richardson, p = 4
        out["richardson"] = {"order_assumed": 4, "J_extrapolated": J_inf,
                             "error_of_N40": (abs(J[40] - J_inf) / abs(J_inf)) if 40 in J else None,
                             "error_of_finest": abs(J2 - J_inf) / abs(J_inf)}
        print(f"\n  Richardson (p=4): J_inf = {J_inf:.6e}; N=40 is {100*abs(J[40]-J_inf)/abs(J_inf):.2f} % off, "
              f"N={meshes[-1]} is {100*abs(J2-J_inf)/abs(J_inf):.3f} % off")
    else:
        print("\n  finest meshes not feasible; no extrapolation")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "collocation_mesh_study.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (FIG_DIR / "collocation_mesh_study.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  written: scripts/figures/collocation_mesh_study.{json,csv}")


if __name__ == "__main__":
    main()
