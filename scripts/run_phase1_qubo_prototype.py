"""Prototype: solve the Phase-1 Earth-escape spiral with the sliding-window QUBO.

De-risking experiment for the "QUBO everywhere, three bodies everywhere" upgrade.
Instead of the closed-form two-body Edelbaum spiral, this raises the orbit from
GEO toward the 200,000 km parking orbit using chained tangential QUBO windows in
the full planar CR3BP (Moon present every step), and reports:

* total binary-thrust Delta-v vs the analytical (1,660 m/s) and honest numeric
  (1,781 m/s) baselines --- i.e. the cost of going fully binary + receding
  horizon;
* number of windows and total time of flight;
* linearisation validity: per-window predicted miss vs true miss, which tells us
  whether the local linearisation holds in the stiff Earth-dominated regime.

Run::

    python -m scripts.run_phase1_qubo_prototype
"""

from __future__ import annotations

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.earth_escape import (
    earth_orbit_apoapsis_periapsis,
    solve_earth_escape_sliding_window,
)
from qalunar.qubo.scheduling_samplers import sample_brute_force
from qalunar.reference.edelbaum import LENGTH_KM, TIME_S, VELOCITY_M_S


# Same Phase-1 parameters as scripts/run_full_mission_demo.py.
EARTH_RADIUS_KM = 6_378.0
R_GEO_KM = 35_786.0
R_HEO_KM = 200_000.0
ACCEL_MAIN_NONDIM = 1.286

# Analytical / numeric baselines from run_phase1_moon_effect.py.
DV_ANALYTICAL_M_S = 1_660.7
DV_NUMERIC_M_S = 1_781.0


def _circular_earth_state(r1: float, mu: float) -> np.ndarray:
    v1 = float(np.sqrt((1.0 - mu) / r1))
    r_syn = np.array([-mu + r1, 0.0])
    v_inertial = np.array([0.0, v1])
    omega_cross_r = np.array([-r_syn[1], r_syn[0]])
    v_syn = v_inertial - omega_cross_r
    return np.array([r_syn[0], r_syn[1], v_syn[0], v_syn[1]])


def main() -> None:
    mu = EARTH_MOON_MU
    r_start = (EARTH_RADIUS_KM + R_GEO_KM) / LENGTH_KM
    r_heo = R_HEO_KM / LENGTH_KM
    dyn = PlanarCR3BP()
    state0 = _circular_earth_state(r_start, mu)

    print("=" * 70)
    print("  Phase-1 Earth-escape via sliding-window QUBO (full CR3BP)")
    print("=" * 70)
    print(f"  start: circular Earth orbit r={r_start:.4f} "
          f"({(EARTH_RADIUS_KM + R_GEO_KM):,.0f} km, GEO)")
    print(f"  target SMA: {r_heo:.4f} ({R_HEO_KM:,.0f} km)")
    print(f"  thrust: {ACCEL_MAIN_NONDIM} nondim (350 mN / 100 kg), binary on/off\n")

    result = solve_earth_escape_sliding_window(
        dyn, state0,
        sampler=lambda q: sample_brute_force(q).schedule,
        target_sma=r_heo,
        n_windows=400,
        window_revs=0.25,
        n_decision_steps=10,
        thrust_magnitude=ACCEL_MAIN_NONDIM,
        boost_factor=1.05,
        target_position_weight=0.0,
        target_velocity_weight=1.0,
        fuel_weight=0.0,
        max_inner_iters=4,
        verbose=True,
    )

    _, _, sma_final = earth_orbit_apoapsis_periapsis(result.final_state, mu=mu)
    r_apo, r_peri, _ = earth_orbit_apoapsis_periapsis(result.final_state, mu=mu)
    dv_qubo_m_s = result.total_delta_v * VELOCITY_M_S
    tof_days = result.total_tof * TIME_S / 86_400.0

    print()
    print("-" * 70)
    print("  Outcome")
    print("-" * 70)
    print(f"  reached target SMA:   {result.reached_target}")
    print(f"  windows used:         {len(result.windows)}")
    print(f"  final SMA:            {sma_final:.4f} "
          f"({sma_final * LENGTH_KM:,.0f} km)")
    print(f"  final apo / peri:     {r_apo * LENGTH_KM:,.0f} km / "
          f"{r_peri * LENGTH_KM:,.0f} km  "
          f"(ecc {(r_apo - r_peri) / (r_apo + r_peri):.3f})")
    print(f"  total burns:          {result.total_burns}")
    print(f"  time of flight:       {tof_days:.2f} days")
    print()
    print(f"  Delta-v (binary QUBO):    {dv_qubo_m_s:>9,.1f} m/s")
    print(f"  Delta-v (numeric spiral): {DV_NUMERIC_M_S:>9,.1f} m/s  "
          f"(penalty {100 * (dv_qubo_m_s / DV_NUMERIC_M_S - 1):+.1f} %)")
    print(f"  Delta-v (analytical):     {DV_ANALYTICAL_M_S:>9,.1f} m/s  "
          f"(penalty {100 * (dv_qubo_m_s / DV_ANALYTICAL_M_S - 1):+.1f} %)")

    # Linearisation validity: predicted vs true per-window miss.
    if result.windows:
        pred = np.array([w.predicted_miss_norm for w in result.windows])
        true = np.array([w.true_miss_norm for w in result.windows])
        rel = np.abs(pred - true) / np.maximum(np.abs(true), 1e-12)
        print()
        print("-" * 70)
        print("  Linearisation validity (per-window predicted vs true miss)")
        print("-" * 70)
        print(f"  median |pred-true|/true:  {np.median(rel):.3e}")
        print(f"  max    |pred-true|/true:  {np.max(rel):.3e}")
        print(f"  windows with >10% gap:    "
              f"{int(np.sum(rel > 0.1))} / {len(result.windows)}")


if __name__ == "__main__":
    main()
