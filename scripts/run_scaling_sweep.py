"""Large-N scaling sweep on the cislunar correction QUBO.

Re-times brute force, MILP (HiGHS via SciPy), simulated annealing,
and Kerberos (dwave-hybrid local) at N = 10, 15, 20, 50, 100, 200.
Persists results to ``scaling_results.json`` for reproducibility
and to feed back into the paper's scaling table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.milp_baseline import solve_scheduling_milp
from qalunar.qubo.scheduling_samplers import (
    sample_brute_force,
    sample_kerberos,
    sample_simulated_annealing,
)
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    build_thrust_scheduling_qubo,
)
from qalunar.reference.edelbaum import LENGTH_KM


_MU = EARTH_MOON_MU
_GM = 1.0 - _MU
_R_HEO = 200_000.0 / LENGTH_KM
_V_CIRC = float(np.sqrt(_GM / _R_HEO))
_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R = np.array([-_R_SYN[1], _R_SYN[0]])
_V_SYN_PERFECT = np.array([0.0, _V_CIRC * 1.190]) - _OMEGA_CROSS_R
_V_SYN_IMPERFECT = np.array([0.0, _V_CIRC * 1.187]) - _OMEGA_CROSS_R
STATE0 = np.concatenate([_R_SYN, _V_SYN_IMPERFECT])
STATE_PERFECT = np.concatenate([_R_SYN, _V_SYN_PERFECT])

THRUST_MAG = 0.02
T_SPAN = (0.0, 3.0)


def build_qubo(N: int):
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_direction="tangential",
        fuel_weight=0.0,
    )
    dyn = PlanarCR3BP()
    _, traj = dyn.propagate(STATE_PERFECT, T_SPAN, n_steps=8000)
    target = traj[-1]
    return build_thrust_scheduling_qubo(
        dyn, STATE0, target, T_SPAN,
        n_decision_steps=N, config=cfg, n_integration_substeps=80,
    )


def main() -> None:
    Ns = [10, 15, 20, 50, 100, 200]
    rows = []

    print(f"{'N':>4} {'method':>10} {'energy':>14} {'time(s)':>10} {'note':>30}")
    print("-" * 75)

    for N in Ns:
        qubo = build_qubo(N)
        row: dict = {"N": N, "results": {}}

        # Brute force: only feasible up to 20
        if N <= 20:
            try:
                t0 = time.perf_counter()
                bf = sample_brute_force(qubo)
                t = time.perf_counter() - t0
                row["results"]["brute_force"] = {
                    "energy": bf.energy, "time": t, "note": "exact",
                }
                print(f"{N:>4} {'BF':>10} {bf.energy:14.4e} {t:>10.3f} "
                      f"{'exact':>30}")
            except Exception as e:
                row["results"]["brute_force"] = {"error": str(e)}
                print(f"{N:>4} {'BF':>10} {'-':>14} {'-':>10} "
                      f"{f'failed: {str(e)[:25]}':>30}")

        # MILP: with 30s time limit at small N, 60s beyond
        time_limit = 30.0 if N <= 20 else 60.0
        try:
            t0 = time.perf_counter()
            milp = solve_scheduling_milp(qubo, time_limit=time_limit, mip_rel_gap=1e-6)
            t = time.perf_counter() - t0
            row["results"]["milp"] = {
                "energy": milp.energy, "time": t,
                "is_optimal": milp.is_optimal, "n_aux": milp.n_aux_vars,
            }
            note = "optimal" if milp.is_optimal else f"timeout {time_limit}s"
            print(f"{N:>4} {'MILP':>10} {milp.energy:14.4e} {t:>10.3f} "
                  f"{note:>30}")
        except Exception as e:
            row["results"]["milp"] = {"error": str(e)}
            print(f"{N:>4} {'MILP':>10} {'-':>14} {'-':>10} "
                  f"{f'failed: {str(e)[:25]}':>30}")

        # SA: fixed 1000 reads
        try:
            t0 = time.perf_counter()
            sa = sample_simulated_annealing(qubo, num_reads=1000, seed=42)
            t = time.perf_counter() - t0
            row["results"]["sa"] = {
                "energy": sa.energy, "time": t, "num_reads": 1000,
            }
            print(f"{N:>4} {'SA':>10} {sa.energy:14.4e} {t:>10.3f} "
                  f"{'1000 reads':>30}")
        except Exception as e:
            row["results"]["sa"] = {"error": str(e)}
            print(f"{N:>4} {'SA':>10} {'-':>14} {'-':>10} "
                  f"{f'failed: {str(e)[:25]}':>30}")

        # Kerberos: local hybrid workflow
        try:
            t0 = time.perf_counter()
            kerb = sample_kerberos(qubo, max_iter=6, convergence=2)
            t = time.perf_counter() - t0
            row["results"]["kerberos"] = {
                "energy": kerb.energy, "time": t,
            }
            print(f"{N:>4} {'Kerberos':>10} {kerb.energy:14.4e} {t:>10.3f} "
                  f"{'local hybrid':>30}")
        except Exception as e:
            row["results"]["kerberos"] = {"error": str(e)}
            print(f"{N:>4} {'Kerberos':>10} {'-':>14} {'-':>10} "
                  f"{f'failed: {str(e)[:80]}':>30}")

        rows.append(row)

    out_path = Path(__file__).resolve().parent / "figures" / "scaling_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
