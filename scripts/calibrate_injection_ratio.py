"""Sweep V_RATIO_INJECTION to find a value that delivers a near-parabolic
perilune state at the end of the Phase-2 corrected flyby.

For an all-binary mission, Phase 3 must be able to capture the spacecraft
without an impulsive pre-capture burn. Capture from a hyperbolic flyby
with low thrust requires the spacecraft to arrive at perilune with
v <= sqrt(2) * v_circ (parabolic threshold). Below that, even a small
amount of anti-tangential thrust during the perilune passage drops the
energy below zero and the orbit is bound.

This script runs Phase 2's iterative QUBO correction for several
candidate ``v_inj / v_circ`` ratios at injection and reports, for each,
the Moon-relative perilune speed in units of the local circular speed.

Run::

    python -m scripts.calibrate_injection_ratio
"""

from __future__ import annotations

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.scheduling_samplers import sample_brute_force
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    propagate_schedule,
    solve_iterative,
)
from qalunar.reference import find_perilune
from qalunar.reference.edelbaum import LENGTH_KM, VELOCITY_M_S


_MU = EARTH_MOON_MU
_R_HEO = 200_000.0 / LENGTH_KM
_OMEGA_CROSS_R = np.array([0.0, _R_HEO - _MU])  # (-y_handoff, x_handoff)
_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R_SYN = np.array([-_R_SYN[1], _R_SYN[0]])
_MOON_POS = np.array([1.0 - _MU, 0.0])

THRUST_MAG = 0.02
T_PHASE2 = 4.0
N_PHASE2 = 15


def evaluate_ratio(v_ratio: float, v_ratio_target: float = 1.190) -> dict:
    """Run Phase 2 with the given injection ratio and report perilune state."""
    dyn = PlanarCR3BP()

    v_circ_local = float(np.sqrt((1.0 - _MU) / _R_HEO))
    state_perfect = np.concatenate([
        _R_SYN,
        np.array([0.0, v_circ_local * v_ratio_target]) - _OMEGA_CROSS_R_SYN,
    ])
    state0 = np.concatenate([
        _R_SYN,
        np.array([0.0, v_circ_local * v_ratio]) - _OMEGA_CROSS_R_SYN,
    ])

    _, traj_perfect = dyn.propagate(state_perfect, (0.0, T_PHASE2),
                                    n_steps=8_000)
    target = traj_perfect[-1]

    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG, thrust_direction="tangential",
        fuel_weight=0.0,
    )
    result = solve_iterative(
        dyn, state0, target, (0.0, T_PHASE2),
        n_decision_steps=N_PHASE2,
        sampler=lambda q: sample_brute_force(q).schedule,
        config=cfg,
        n_integration_substeps=80, n_truth_substeps=200, max_iters=8,
    )

    _, traj = propagate_schedule(
        dyn, state0, (0.0, T_PHASE2), result.schedule, cfg,
        n_integration_substeps=300, return_trajectory=True,
    )
    dist = np.linalg.norm(traj[:, :2] - _MOON_POS, axis=1)
    idx = int(np.argmin(dist))
    peri = find_perilune(dyn, traj[idx], (0.0, 0.05), n_steps=2_000)

    v_circ_at_peri = peri.v_circ
    ratio_at_peri = peri.v_perilune / v_circ_at_peri
    return {
        "v_ratio": v_ratio,
        "v_ratio_target": v_ratio_target,
        "phase2_dV_m_s": result.final_qubo.delta_v(result.schedule)
                         * VELOCITY_M_S,
        "miss_km": float(np.linalg.norm(result.true_miss[:2])) * LENGTH_KM,
        "perilune_alt_km": (peri.r_perilune * LENGTH_KM - 1737.0),
        "v_perilune_m_s": peri.v_perilune * VELOCITY_M_S,
        "v_circ_perilune_m_s": v_circ_at_peri * VELOCITY_M_S,
        "v_over_circ": ratio_at_peri,
        "is_bound": peri.is_bound,
    }


def main() -> None:
    print(f"{'v_ratio':>10} {'target':>8} {'P2_dV':>9} {'miss(km)':>10} "
          f"{'peri_alt':>10} {'v_peri':>10} {'v/v_circ':>10} {'bound?':>8}")
    print("-" * 88)
    candidates = [
        (1.190, 1.190),
        (1.187, 1.190),
        (1.182, 1.182),
        (1.180, 1.180),
        (1.178, 1.178),
        (1.175, 1.175),
        (1.170, 1.170),
        (1.165, 1.165),
    ]
    for v_inj, v_target in candidates:
        try:
            r = evaluate_ratio(v_inj, v_target)
            print(
                f"{r['v_ratio']:>10.4f} {r['v_ratio_target']:>8.3f} "
                f"{r['phase2_dV_m_s']:>9.1f} {r['miss_km']:>10.0f} "
                f"{r['perilune_alt_km']:>10.0f} "
                f"{r['v_perilune_m_s']:>10.0f} "
                f"{r['v_over_circ']:>10.3f} "
                f"{'yes' if r['is_bound'] else 'no':>8}"
            )
        except Exception as e:
            print(f"{v_inj:>10.4f} -> error: {e}")

    print()
    print("Capture threshold: v/v_circ <= sqrt(2) ~ 1.414  =>  bound")
    print("Want a value just below 1.414 (slightly negative energy)")
    print("so the QUBO can finalise capture in Phase 3.")


if __name__ == "__main__":
    main()
