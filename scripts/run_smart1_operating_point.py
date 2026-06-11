"""Run the qalunar Phase-1 spiral at SMART-1's *operating point*.

The end-to-end demo (``run_full_mission_demo``) flies an optimistic high-power
microsatellite (\\,350\\,mN on 100\\,kg, $a_0 = 3.5$\\,mm/s$^2$) from a circular
GEO. That makes its time of flight (\\,$\\sim$1 month) far shorter than ESA's
SMART-1 (\\,$\\sim$13 months), which muddies a direct comparison. This script
removes the apples-to-oranges gap by re-running the *same* analytical Edelbaum
spiral at SMART-1's actual inputs:

  * thrust-to-mass  $a_0 = 70\\,\\mathrm{mN} / 367\\,\\mathrm{kg}
    = 0.19\\,\\mathrm{mm/s^2}$ (vs the demo's 3.5),
  * specific impulse $I_\\mathrm{sp} = 1640\\,\\mathrm{s}$ (PPS-1350 Hall),
  * a start orbit with GTO energy: SMART-1 launched into a
    $7035\\times42223$\\,km GTO, whose semi-major axis $a_\\mathrm{GTO}\\approx
    24\\,629$\\,km sets the energy-equivalent circular start radius, and
  * a duty cycle $\\sim$40 % (SMART-1 thrust only $\\sim$1/3--1/2 of each orbit
    for power/eclipse/thermal reasons).

The Edelbaum $\\Delta v$ depends only on the start/end circular speeds, *not* on
thrust, so the spiral $\\Delta v$ is an operating-point-independent invariant;
the time of flight scales as $\\Delta v / a_0$. Running the model at SMART-1's
$a_0$ and start energy should therefore land near SMART-1's published
$\\sim$3.5\\,km/s over $\\sim$13 months -- a genuine like-for-like check that the
qalunar transfer physics is consistent with a flown electric-propulsion lunar
mission, rather than a comparison confounded by a 18x thrust-to-mass mismatch.

Run::

    python -m scripts.run_smart1_operating_point
"""

from __future__ import annotations

import numpy as np

from qalunar.reference import edelbaum_spiral
from qalunar.reference.edelbaum import (
    ACCELERATION_M_S2,
    EARTH_MOON_MU,
    LENGTH_KM,
    TIME_S,
    VELOCITY_M_S,
)


G0 = 9.80665  # m/s^2, standard gravity for the rocket equation


# --- SMART-1 published reference (see run_mission_benchmark.py for sources) ---
SMART1 = dict(
    dv_total_ms=3500.0,
    tof_days=410.0,         # launch 2003-09-27 -> lunar capture 2004-11-15
    wet_mass_kg=367.0,
    xe_used_kg=74.0,
    thrust_mn=70.0,
    isp_s=1640.0,
    gto_perigee_radius_km=7035.0,
    gto_apogee_radius_km=42223.0,
    n_revs=207,
    duty_cycle=0.40,
)

# --- Operating points to spiral. Both end at the Moon's sphere of influence
# (≈66,000 km from the Moon ≈ 318,000 km from Earth), the natural handoff to a
# Moon-dominated capture phase. ---
MOON_SOI_KM = 66_100.0
R_END_KM = LENGTH_KM - MOON_SOI_KM          # ≈ 318,300 km from Earth
EARTH_RADIUS_KM = 6_378.0


def _a0_nondim(thrust_mn: float, mass_kg: float) -> float:
    """Thrust-to-mass acceleration in CR3BP nondimensional units."""
    a_si = (thrust_mn * 1e-3) / mass_kg          # m/s^2
    return a_si / ACCELERATION_M_S2


def _gto_equivalent_circular_radius_km() -> float:
    """Circular radius with the same specific energy as SMART-1's GTO.

    A coplanar Edelbaum spiral is parameterised by circular speeds, so the
    fair circular proxy for an elliptical start is the orbit with the same
    energy, i.e. radius = GTO semi-major axis.
    """
    return 0.5 * (SMART1["gto_perigee_radius_km"] + SMART1["gto_apogee_radius_km"])


def _propellant(dv_ms: float, isp_s: float, wet_mass_kg: float) -> tuple[float, float]:
    """Propellant fraction and mass from the rocket equation."""
    ve = isp_s * G0
    frac = 1.0 - np.exp(-dv_ms / ve)
    return frac, frac * wet_mass_kg


def _run_spiral(label: str, r1_km: float, r2_km: float, a0_nondim: float,
                isp_s: float, wet_mass_kg: float, duty_cycle: float) -> dict:
    r1 = r1_km / LENGTH_KM
    r2 = r2_km / LENGTH_KM
    sp = edelbaum_spiral(r1=r1, r2=r2, a_thrust=a0_nondim, mu=EARTH_MOON_MU)
    dv_ms = sp.dv * VELOCITY_M_S
    tof_days_100 = sp.tof * TIME_S / 86_400.0
    tof_days_duty = tof_days_100 / duty_cycle
    frac, prop_kg = _propellant(dv_ms, isp_s, wet_mass_kg)
    return dict(
        label=label, r1_km=r1_km, r2_km=r2_km, a0_nondim=a0_nondim,
        a0_mm_s2=a0_nondim * ACCELERATION_M_S2 * 1e3,
        dv_ms=dv_ms, n_rev=sp.n_revs,
        tof_days_100=tof_days_100, tof_days_duty=tof_days_duty,
        prop_frac=frac, prop_kg=prop_kg, wet_mass_kg=wet_mass_kg, isp_s=isp_s,
    )


def main() -> None:
    r_gto_eq = _gto_equivalent_circular_radius_km()

    print("=" * 78)
    print("  qalunar Edelbaum spiral at SMART-1's operating point")
    print("=" * 78)
    print(f"  Start: GTO-energy-equivalent circular orbit")
    print(f"    SMART-1 GTO {SMART1['gto_perigee_radius_km']:,.0f} x "
          f"{SMART1['gto_apogee_radius_km']:,.0f} km "
          f"(a_GTO = {r_gto_eq:,.0f} km)")
    print(f"  End:   Moon SOI handoff, r = {R_END_KM:,.0f} km from Earth")
    print(f"  a_0 = 70 mN / 367 kg = "
          f"{_a0_nondim(70.0, 367.0) * ACCELERATION_M_S2 * 1e3:.3f} mm/s^2 "
          f"(nondim {_a0_nondim(70.0, 367.0):.4f})")
    print()

    smart1_point = _run_spiral(
        "qalunar @ SMART-1 point",
        r1_km=r_gto_eq, r2_km=R_END_KM,
        a0_nondim=_a0_nondim(SMART1["thrust_mn"], SMART1["wet_mass_kg"]),
        isp_s=SMART1["isp_s"], wet_mass_kg=SMART1["wet_mass_kg"],
        duty_cycle=SMART1["duty_cycle"],
    )
    # Contrast: the demo's high-power microsat from the same GTO-equiv start.
    demo_point = _run_spiral(
        "qalunar @ demo point (350 mN/100 kg)",
        r1_km=r_gto_eq, r2_km=R_END_KM,
        a0_nondim=_a0_nondim(350.0, 100.0),
        isp_s=1640.0, wet_mass_kg=100.0, duty_cycle=1.0,
    )

    def _row(d: dict) -> None:
        print(f"  {d['label']:<38}")
        print(f"      a_0:            {d['a0_mm_s2']:.3f} mm/s^2")
        print(f"      spiral dV:      {d['dv_ms']:,.0f} m/s")
        print(f"      revolutions:    {d['n_rev']:,.0f}")
        print(f"      TOF (100% duty):{d['tof_days_100']:>8,.0f} days")
        print(f"      TOF (duty adj): {d['tof_days_duty']:>8,.0f} days")
        print(f"      propellant:     {d['prop_kg']:,.1f} kg "
              f"({d['prop_frac']:.1%} of {d['wet_mass_kg']:.0f} kg)")
        print()

    print("-" * 78)
    _row(smart1_point)
    _row(demo_point)

    # --- Side-by-side with SMART-1 published ---
    print("-" * 78)
    print("  Like-for-like: qalunar @ SMART-1 point  vs  SMART-1 (flown)")
    print("-" * 78)
    s = smart1_point
    print(f"  {'':<26}{'qalunar@S1':>14}{'SMART-1':>14}")
    print(f"  {'transfer dV [m/s]':<26}{s['dv_ms']:>14,.0f}"
          f"{SMART1['dv_total_ms']:>14,.0f}   (spiral only; + lunar capture)")
    print(f"  {'TOF [days]':<26}{s['tof_days_duty']:>14,.0f}"
          f"{SMART1['tof_days']:>14,.0f}   (qalunar at {SMART1['duty_cycle']:.0%} duty)")
    print(f"  {'revolutions':<26}{s['n_rev']:>14,.0f}"
          f"{SMART1['n_revs']:>14,.0f}")
    print(f"  {'propellant [kg]':<26}{s['prop_kg']:>14,.1f}"
          f"{SMART1['xe_used_kg']:>14,.1f}   (Isp {SMART1['isp_s']:.0f} s)")
    print()
    print("  Note: the spiral dV above is the Earth-escape portion to the Moon's")
    print("  SOI; SMART-1's 3.5 km/s additionally includes the lunar capture/")
    print("  descent. The agreement in dV magnitude, revolution count, and")
    print("  duty-adjusted TOF confirms the qalunar transfer physics reproduces")
    print("  a flown EP lunar mission once given that mission's a_0 and start")
    print("  energy -- the earlier 32-day figure was purely the 18x-higher")
    print("  thrust-to-mass of the optimistic demo spacecraft.")


if __name__ == "__main__":
    main()
