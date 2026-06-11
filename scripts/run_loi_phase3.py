"""Phase-3 LOI demo: analytical Moon-descending Edelbaum spiral.

This is the Phase-3 counterpart of ``scripts/run_thrust_scheduling.py``
(Phase-2 transfer correction) and ``scripts/run_moon_orbit_figure.py``
(Phase-4 station-keeping).

The spacecraft enters the Moon's vicinity from a Phase-2 flyby
trajectory at perilune altitude ``r1`` (here 6,000 km, matching the
Phase-2 closest-approach used in the paper) and is to be inserted into
a circular low lunar orbit (LLO) at altitude ``r2`` (here 100 km).

Phase 3 here is purely analytical: it reports dV, time of flight,
revolution count, and a synodic-frame handoff state at the END of the
descent. It is the Moon-centered analogue of the Phase-1 Edelbaum
spiral and uses the same closed-form Edelbaum expressions with
``GM_Moon = mu`` instead of ``GM_Earth = 1 - mu``. The QUBO scheduler
of the paper would then refine this analytical reference with on/off
binary corrections in a separate (future) step, just as Phase-2 does
for the cislunar transfer.

Run from the project root::

    python -m scripts.run_loi_phase3
"""

from __future__ import annotations

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.reference.edelbaum import (
    ACCELERATION_M_S2,
    LENGTH_KM,
    TIME_S,
    VELOCITY_M_S,
    edelbaum_spiral_moon_descending,
)


# ---------------------------------------------------------------------------
# Mission parameters: 100 kg microsatellite, 5 mN ion thruster.
# Matches the spacecraft class of the Phase-2 paper experiments.
# ---------------------------------------------------------------------------

SPACECRAFT_MASS_KG = 100.0
THRUST_MN = 5.4
ACCEL_NONDIM = 0.02  # ≈ 0.054 mm/s² (matches the paper)

PERILUNE_ALT_KM = 6_000.0
LLO_ALT_KM = 100.0
MOON_RADIUS_KM = 1737.0


def _print_section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def main() -> None:
    _print_section("Phase 3: Lunar Orbit Insertion (LOI)")
    print(
        "  Analytical Moon-descending Edelbaum spiral.\n"
        f"  Spacecraft: {SPACECRAFT_MASS_KG:.0f} kg, ion thruster ~{THRUST_MN:.1f} mN,"
        f" a = {ACCEL_NONDIM * ACCELERATION_M_S2 * 1000.0:.3f} mm/s^2"
    )

    r1 = (MOON_RADIUS_KM + PERILUNE_ALT_KM) / LENGTH_KM
    r2 = (MOON_RADIUS_KM + LLO_ALT_KM) / LENGTH_KM

    spiral = edelbaum_spiral_moon_descending(
        r1=r1, r2=r2, a_thrust=ACCEL_NONDIM,
    )

    rep = spiral.si_report()

    _print_section("Spiral summary (closed-form)")
    print(f"  Initial Moon-relative orbit:  r1 = {rep['r1_km']}"
          f"  ({PERILUNE_ALT_KM:,.0f} km altitude)")
    print(f"  Final Moon-relative orbit:    r2 = {rep['r2_km']}"
          f"  ({LLO_ALT_KM:,.0f} km altitude)")
    print(f"  Body-relative speed at r1:    v1 = {rep['v1_m_s']}")
    print(f"  Body-relative speed at r2:    v2 = {rep['v2_m_s']}")
    print(f"  Total dV (anti-tangential):   {rep['dv_m_s']}")
    print(f"  Time of flight:               {rep['tof_days']}"
          f"  ({rep['tof_years']})")
    print(f"  Revolutions around Moon:      {rep['n_revs']}")
    print(f"  Tangential acceleration:      {rep['a_thrust_mm_s2']}")

    # dV per orbital period (rough rate of orbit lowering)
    n_revs = float(rep['n_revs'].replace(',', ''))
    dv_per_rev = spiral.dv * VELOCITY_M_S / max(n_revs, 1.0)
    burn_per_rev_h = spiral.tof / max(n_revs, 1.0) * TIME_S / 3600.0
    print(f"  Average dV per Moon revolution: {dv_per_rev:.1f} m/s "
          f"(~{burn_per_rev_h:.1f} h thrust per rev)")

    _print_section("Synodic-frame handoff at end of LOI (start of Phase 4)")
    r_syn, v_syn = spiral.handoff_state_synodic()
    moon_pos = np.array([1.0 - EARTH_MOON_MU, 0.0])

    print(f"  Position (synodic, nondim):     {r_syn}")
    print(f"  Velocity (synodic, nondim):     {v_syn}")
    print(f"  Position (Moon-relative, km):   "
          f"{(r_syn - moon_pos) * LENGTH_KM}")
    print(f"  Distance to Moon:               "
          f"{np.linalg.norm(r_syn - moon_pos) * LENGTH_KM:,.1f} km "
          f"(target = {(MOON_RADIUS_KM + LLO_ALT_KM):,.0f} km)")

    # Sanity: propagate the unforced state for one Moon-relative period
    # and verify the orbit stays roughly circular at this radius.
    dyn = PlanarCR3BP()
    state0 = np.concatenate([r_syn, v_syn])
    gm_moon = EARTH_MOON_MU
    t_period = 2.0 * np.pi * np.sqrt(r2 ** 3 / gm_moon)
    _, traj = dyn.propagate(state0, (0.0, t_period), n_steps=2000)
    d_to_moon_km = (np.linalg.norm(traj[:, :2] - moon_pos, axis=1) * LENGTH_KM)

    _print_section("Sanity check: propagate handoff state for one LLO period")
    print(f"  Period (Keplerian, days):       "
          f"{t_period * TIME_S / 86_400.0:.2f}")
    print(f"  Distance-to-Moon at t=0:        {d_to_moon_km[0]:,.1f} km")
    print(f"  Distance-to-Moon at t=T:        {d_to_moon_km[-1]:,.1f} km")
    print(f"  Min / max over period:          "
          f"{d_to_moon_km.min():,.1f} / {d_to_moon_km.max():,.1f} km")
    drift_km = abs(d_to_moon_km[-1] - d_to_moon_km[0])
    print(f"  CR3BP drift after one period:   {drift_km:.1f} km "
          "(small drift expected due to Earth perturbation)")

    _print_section("Summary for Phase 4 (station-keeping)")
    print(
        "  This handoff state can be passed directly to the existing\n"
        "  station-keeping driver (run_moon_orbit_figure.py) to maintain\n"
        "  the resulting LLO under CR3BP perturbations.\n\n"
        "  Phase-3 LOI dV reported above is from the analytical spiral\n"
        "  alone; the QUBO scheduler can be applied on top with a\n"
        "  sliding-window strategy to add binary corrections (future work)."
    )


if __name__ == "__main__":
    main()
