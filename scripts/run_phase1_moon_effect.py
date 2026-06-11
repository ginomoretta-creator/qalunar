"""Quantify the Moon's effect on the Phase-1 Earth-escape spiral.

Phase 1 is normally a closed-form, Moon-free Edelbaum spiral
(:func:`qalunar.reference.edelbaum.edelbaum_spiral`). This script runs the same
spiral three ways and isolates the lunar third-body perturbation:

1. **Closed-form** Edelbaum (Moon ignored, quasi-circular averaging) --- the
   analytical reference.
2. **Numeric, Moon OFF** --- the same RK4 integrator and tangential-thrust
   convention as (3), but with the lunar gravity term removed. This is the
   honest Moon-free baseline; differences from (1) are the cost of the
   averaging approximation, *not* the Moon.
3. **Numeric, Moon ON** --- the full planar CR3BP.

The Moon's true effect is (3) - (2): same integrator, same termination, only
the lunar gravity term toggled. Comparing (3) against (1) instead would
conflate the Moon with the averaging error and with orbital phase.

Run::

    python -m scripts.run_phase1_moon_effect
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics import PlanarCR3BP
from qalunar.reference import edelbaum_spiral, phase1_spiral_numeric
from qalunar.reference.edelbaum import LENGTH_KM, TIME_S, VELOCITY_M_S


class _NoMoonCR3BP(PlanarCR3BP):
    """Rotating-frame dynamics identical to the CR3BP but with the Moon's
    gravity term removed.

    Keeps the Coriolis term, the centrifugal term and the Earth gravity term
    exactly as in :class:`PlanarCR3BP`, dropping only the ``mu / r2**3`` lunar
    attraction. This is the correct Moon-free baseline for isolating the lunar
    perturbation: everything except the Moon's gravity is byte-for-byte the
    same as the full model, so a difference in the propagated trajectory is
    attributable to the Moon alone.
    """

    def rhs(
        self,
        state: NDArray[np.float64],
        control: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        x, y, vx, vy = state
        r1 = float(np.sqrt((x + self.mu) ** 2 + y ** 2))
        one_minus_mu = 1.0 - self.mu
        omega_x = x - one_minus_mu * (x + self.mu) / r1 ** 3
        omega_y = y - one_minus_mu * y / r1 ** 3
        ax = 2.0 * vy + omega_x
        ay = -2.0 * vx + omega_y
        if control is not None:
            ax += control[0]
            ay += control[1]
        return np.array([vx, vy, ax, ay])


# Same Phase-1 parameters as scripts/run_full_mission_demo.py.
EARTH_RADIUS_KM = 6_378.0
R_GEO_KM = 35_786.0
R_HEO_KM = 200_000.0
ACCEL_MAIN_NONDIM = 1.286    # 350 mN / 100 kg = 3.5 mm/s^2


def main() -> None:
    r_start = (EARTH_RADIUS_KM + R_GEO_KM) / LENGTH_KM
    r_heo = R_HEO_KM / LENGTH_KM

    print("=" * 70)
    print("  Phase-1 Earth-escape spiral: Moon-free vs Moon-included")
    print("=" * 70)
    print(f"  r1 = {r_start:.4f}  ({(EARTH_RADIUS_KM + R_GEO_KM):,.0f} km, GEO)")
    print(f"  r2 = {r_heo:.4f}  ({R_HEO_KM:,.0f} km parking orbit)")
    print(f"  a_thrust = {ACCEL_MAIN_NONDIM} nondim (350 mN / 100 kg)\n")

    # (1) Closed-form, Moon ignored (quasi-circular averaging).
    spiral = edelbaum_spiral(r1=r_start, r2=r_heo, a_thrust=ACCEL_MAIN_NONDIM)

    # (2) Numeric baseline with the Moon's gravity term removed.
    num_off = phase1_spiral_numeric(
        r1=r_start, r2=r_heo, a_thrust=ACCEL_MAIN_NONDIM,
        dynamics=_NoMoonCR3BP(),
    )

    # (3) Numeric, full CR3BP (Moon included).
    dyn = PlanarCR3BP()
    num_on = phase1_spiral_numeric(
        r1=r_start, r2=r_heo, a_thrust=ACCEL_MAIN_NONDIM, dynamics=dyn,
    )

    def row(label: str, a: float, b: float, c: float, unit: str) -> None:
        print(f"  {label:<14}{a:>13,.3f}{b:>13,.3f}{c:>13,.3f}   {unit}")

    print(f"  {'':<14}{'closed-form':>13}{'num Moon-off':>13}"
          f"{'num Moon-on':>13}")
    row("TOF", spiral.tof * TIME_S / 86_400.0,
        num_off.tof * TIME_S / 86_400.0,
        num_on.tof * TIME_S / 86_400.0, "days")
    row("revolutions", spiral.n_revs, num_off.n_revs, num_on.n_revs, "revs")
    row("dV", spiral.dv * VELOCITY_M_S,
        num_off.dv * VELOCITY_M_S,
        num_on.dv * VELOCITY_M_S, "m/s")
    if not (num_off.converged and num_on.converged):
        print("\n  WARNING: a numeric spiral did not reach r2.")

    # --- The Moon's true effect: (3) - (2), same integrator/termination ---
    dpos = np.linalg.norm(
        num_on.handoff_state[:2] - num_off.handoff_state[:2]) * LENGTH_KM
    dvel = np.linalg.norm(
        num_on.handoff_state[2:4] - num_off.handoff_state[2:4]) * VELOCITY_M_S
    dtof_h = (num_on.tof - num_off.tof) * TIME_S / 3600.0

    print()
    print("-" * 70)
    print("  Lunar perturbation at handoff  (num Moon-on  -  num Moon-off)")
    print("-" * 70)
    print(f"  Position shift: {dpos:>12,.1f} km")
    print(f"  Velocity shift: {dvel:>12,.2f} m/s")
    print(f"  Extra TOF:      {dtof_h:>12,.2f} h")
    print()
    moon_dist_km = (1.0 - dyn.mu) * LENGTH_KM - num_on.r2 * LENGTH_KM
    print(f"  (Handoff sits ~{num_on.r2 * LENGTH_KM:,.0f} km from Earth, i.e. "
          f"~{moon_dist_km:,.0f} km from the Moon.)")


if __name__ == "__main__":
    main()
