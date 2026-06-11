"""Try submitting the scheduling QUBO to D-Wave Leap Hybrid.

Detects whether a Leap token is configured. If yes, submits a small
benchmark sweep (N=20, 40, 60) and reports time-to-solution and
solution quality vs simulated annealing. If no, prints clear setup
instructions and exits 0.

Usage::

    # configure once
    export DWAVE_API_TOKEN=...
    # or: dwave config create

    python -m scripts.try_leap_hybrid
"""

from __future__ import annotations

import sys

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.scheduling_samplers import (
    leap_token_available,
    sample_leap_hybrid,
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

V_RATIO = 1.187
THRUST_MAG = 0.02
T_SPAN = (0.0, 3.0)


def _state(v_ratio: float) -> np.ndarray:
    return np.concatenate([
        _R_SYN,
        np.array([0.0, _V_CIRC * v_ratio]) - _OMEGA_CROSS_R,
    ])


def _build(N: int, dyn: PlanarCR3BP):
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_direction="tangential",
        fuel_weight=0.0,
    )
    state0 = _state(V_RATIO)
    state_perfect = _state(1.190)
    _, traj = dyn.propagate(state_perfect, T_SPAN, n_steps=8000)
    target = traj[-1]
    return build_thrust_scheduling_qubo(
        dyn, state0, target, T_SPAN,
        n_decision_steps=N, config=cfg, n_integration_substeps=80,
    )


def main() -> int:
    if not leap_token_available():
        print("No Leap API token detected.")
        print()
        print("To configure:")
        print("  1. Sign in at https://cloud.dwavesys.com/leap/")
        print("  2. Copy your API token from your account profile.")
        print("  3. Either:")
        print("     export DWAVE_API_TOKEN=<your-token>")
        print("     or run: dwave config create")
        print()
        print("Falling back to simulated annealing for a smoke test.")
        dyn = PlanarCR3BP()
        qubo = _build(N=15, dyn=dyn)
        sa = sample_simulated_annealing(qubo, num_reads=500)
        print(f"  SA  on N=15: energy={sa.energy:.4e}, time={sa.solve_time:.2f}s")
        return 0

    print("Leap token detected. Running benchmark sweep.\n")
    dyn = PlanarCR3BP()

    print(f"{'N':>4} {'method':>15} {'energy':>14} {'time(s)':>10} {'qpu(us)':>10}")
    print("-" * 60)
    for N in (20, 40, 60):
        qubo = _build(N=N, dyn=dyn)

        sa = sample_simulated_annealing(qubo, num_reads=1000)
        print(f"{N:4d} {'SA':>15} {sa.energy:14.4e} {sa.solve_time:10.3f} "
              f"{'-':>10}")

        try:
            hyb = sample_leap_hybrid(qubo, label=f"qalunar-N{N}")
        except Exception as exc:
            print(f"{N:4d} {'leap_hybrid':>15} -- error: {exc}")
            continue
        qpu = hyb.metadata.get("qpu_access_time") or 0
        print(f"{N:4d} {'leap_hybrid':>15} {hyb.energy:14.4e} "
              f"{hyb.solve_time:10.3f} {qpu:10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
