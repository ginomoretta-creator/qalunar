"""End-to-end Earth-Moon mission, all-binary thrust scheduling.

Single high-thrust ion/Hall propulsion system (~350 mN on a 100 kg
microsatellite) drives every phase of the transfer with on/off binary
commands. No impulsive burns: the cislunar injection and the lunar
capture are absorbed into continuous-thrust arcs and sliding-window
QUBO scheduling.

    Phase 1 -- Edelbaum Earth-ascending spiral + injection boost
               (analytical: spiral to parking orbit, then continuous
               tangential thrust until cislunar injection velocity)
    Phase 2 -- cislunar transfer with QUBO correction
    Phase 3 -- sliding-window QUBO lunar capture and stabilisation
               (no separate pre-capture burn -- the first windows of
               Phase 3 brake the spacecraft below escape velocity at
               perilune by accumulating short anti-tangential pulses
               within the close-approach interval)

Run::

    python -m scripts.run_full_mission_demo
"""

from __future__ import annotations

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.lunar_capture import (
    moon_orbit_apolune_perilune,
    moon_two_body_energy,
    solve_lunar_capture_sliding_window,
)
from qalunar.qubo.scheduling_samplers import (
    sample_brute_force,
    sample_simulated_annealing,
)
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    propagate_schedule,
    solve_iterative,
)
from qalunar.reference import (
    edelbaum_spiral,
    find_perilune,
)
from qalunar.reference.edelbaum import (
    ACCELERATION_M_S2,
    LENGTH_KM,
    TIME_S,
    VELOCITY_M_S,
)


# ---------------------------------------------------------------------------
# All-binary mission parameters
# ---------------------------------------------------------------------------

# Single ion/Hall propulsion system with throttling. The same thruster
# operates at TWO power levels in different mission phases:
#
#   * MAIN power (350 mN): Phase 1 spiral, injection boost, Phase 3
#     capture/stabilisation. Drives the bulk dV cheaply and -- crucially
#     -- gives enough dV during the perilune passage of the hyperbolic
#     flyby (~63 m/s in 5 hours) to brake below escape velocity.
#
#   * CRUISE power (5.4 mN): Phase 2 cislunar trajectory correction. The
#     correction needs only ~16 m/s of dV across long arcs, and high
#     thrust would make each binary slot far too coarse. Throttling down
#     gives precise, fine-grained control while still using the same
#     propulsion system. Real ion/Hall thrusters (NEXT-C, BepiColombo
#     T6, SPT-100) are designed to throttle across this range.
SPACECRAFT_MASS_KG = 100.0
THRUST_MAIN_MN = 350.0
THRUST_CRUISE_MN = 5.4
ACCEL_MAIN_NONDIM = 1.286    # 3.5 mm/s² (350 mN / 100 kg)
ACCEL_CRUISE_NONDIM = 0.02   # 0.054 mm/s² (5.4 mN / 100 kg)

# Phase 1: GEO -> parking orbit, then injection boost.
R_GEO_KM = 35_786.0
R_PHASE1_START_KM = R_GEO_KM
R_HEO_KM = 200_000.0

# Phase 2: cislunar transfer (QUBO correction).
V_RATIO_PERFECT = 1.190
V_RATIO_INJECTION = 1.187
T_PHASE2 = 4.0
N_PHASE2 = 15    # matches the paper's hero scenario at cruise thrust

# Phase 3: lunar capture + stabilisation (sliding-window QUBO).
LLO_ALT_KM = 100.0

EARTH_RADIUS_KM = 6_378.0
MOON_RADIUS_KM = 1_737.0


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _phase_header(idx: str, title: str) -> None:
    print()
    print(f"---- Phase {idx}: {title} ".ljust(70, "-"))


def main() -> None:
    _section("Earth-Moon mission demo: ALL-BINARY thrust scheduling")
    print(
        f"  Spacecraft: {SPACECRAFT_MASS_KG:.0f} kg microsatellite, "
        f"throttleable ion/Hall thruster\n"
        f"  Main power:   {THRUST_MAIN_MN:.0f} mN -> "
        f"a = {ACCEL_MAIN_NONDIM * ACCELERATION_M_S2 * 1000.0:.3f} mm/s^2\n"
        f"  Cruise power: {THRUST_CRUISE_MN:.1f} mN -> "
        f"a = {ACCEL_CRUISE_NONDIM * ACCELERATION_M_S2 * 1000.0:.3f} mm/s^2\n"
        f"  Phase-1 starts at GEO ({R_PHASE1_START_KM:,.0f} km alt)\n"
        f"  All-binary mission: no impulsive burns"
    )

    dyn = PlanarCR3BP()

    # -----------------------------------------------------------------
    # Phase 1: Edelbaum spiral + injection boost (continuous tangential)
    # -----------------------------------------------------------------
    _phase_header("1", "Edelbaum spiral + cislunar injection boost")

    r_start = (EARTH_RADIUS_KM + R_PHASE1_START_KM) / LENGTH_KM
    r_heo = R_HEO_KM / LENGTH_KM
    spiral1 = edelbaum_spiral(r1=r_start, r2=r_heo, a_thrust=ACCEL_MAIN_NONDIM)
    rep1 = spiral1.si_report()

    # State on parking orbit at end of analytical Edelbaum spiral
    r_handoff_1, _ = spiral1.handoff_state_synodic()

    # Cislunar injection: continuous tangential thrust extends Phase 1
    # by an additional dV_inj of velocity gain. With binary "all-on"
    # thrust this is just an additional boost arc on the same thruster.
    omega_cross_r1 = np.array([-r_handoff_1[1], r_handoff_1[0]])
    v_circ_local = float(np.sqrt((1.0 - EARTH_MOON_MU) / r_heo))
    dv_injection_nondim = float(
        v_circ_local * (V_RATIO_INJECTION - 1.0)
    )
    tof_inject_nondim = dv_injection_nondim / ACCEL_MAIN_NONDIM
    dv_phase1_total_nondim = spiral1.dv + dv_injection_nondim
    tof_phase1_total_nondim = spiral1.tof + tof_inject_nondim

    print(f"  GEO ({R_PHASE1_START_KM:,.0f} km alt) "
          f"-> {R_HEO_KM:,.0f} km parking orbit + injection boost")
    print(f"    Spiral dV:        {rep1['dv_m_s']}")
    print(f"    Spiral TOF:       {rep1['tof_days']}")
    print(f"    Injection boost:  "
          f"{dv_injection_nondim * VELOCITY_M_S:.0f} m/s "
          f"({tof_inject_nondim * TIME_S / 3600.0:.1f} h continuous)")
    print(f"    PHASE-1 TOTAL dV: "
          f"{dv_phase1_total_nondim * VELOCITY_M_S:,.0f} m/s")
    print(f"    PHASE-1 TOTAL TOF:"
          f" {tof_phase1_total_nondim * TIME_S / 86_400.0:,.1f} days")

    # State at the end of Phase 1 (after Edelbaum + injection boost):
    # same parking-orbit position, velocity raised to v_inj.
    state_perfect = np.concatenate([
        r_handoff_1,
        np.array([0.0, v_circ_local * V_RATIO_PERFECT]) - omega_cross_r1,
    ])
    state_after_phase1 = np.concatenate([
        r_handoff_1,
        np.array([0.0, v_circ_local * V_RATIO_INJECTION]) - omega_cross_r1,
    ])

    # -----------------------------------------------------------------
    # Phase 2: Cislunar QUBO correction
    # -----------------------------------------------------------------
    _phase_header("2", "Cislunar transfer with binary QUBO correction")

    _, traj_perfect = dyn.propagate(state_perfect, (0.0, T_PHASE2),
                                    n_steps=8_000)
    target_phase2 = traj_perfect[-1]

    # Throttle the thruster down to cruise power for fine-grained Phase-2
    # corrections. Same hardware, smaller effective dV per binary slot.
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=ACCEL_CRUISE_NONDIM,
        thrust_direction="tangential",
        fuel_weight=0.0,
    )

    result = solve_iterative(
        dyn, state_after_phase1, target_phase2, (0.0, T_PHASE2),
        n_decision_steps=N_PHASE2,
        sampler=lambda q: sample_brute_force(q).schedule,
        config=cfg,
        n_integration_substeps=80, n_truth_substeps=300, max_iters=8,
    )
    schedule = result.schedule
    n_burns = int(schedule.sum())
    dv_phase2_nondim = float(result.final_qubo.delta_v(schedule))
    miss_km = float(np.linalg.norm(result.true_miss[:2])) * LENGTH_KM
    print(f"  Iterations:           {result.iterations}")
    print(f"  Termination:          {result.converged_reason}")
    print(f"  Burns:                {n_burns} / {N_PHASE2}")
    print(f"  dV_2 (corrections):   "
          f"{dv_phase2_nondim * VELOCITY_M_S:.1f} m/s")
    print(f"  Final position miss:  {miss_km:,.1f} km")

    # -----------------------------------------------------------------
    # Phase 3: All-binary lunar capture + stabilisation
    # -----------------------------------------------------------------
    _phase_header("3", "Sliding-window QUBO lunar capture (binary)")

    # Find perilune of the corrected flyby trajectory.
    state_phase2_end = result.true_final_state
    moon_pos = np.array([1.0 - EARTH_MOON_MU, 0.0])
    _, traj_phase2 = propagate_schedule(
        dyn, state_after_phase1, (0.0, T_PHASE2), schedule, cfg,
        n_integration_substeps=300, return_trajectory=True,
    )
    dist_phase2 = np.linalg.norm(traj_phase2[:, :2] - moon_pos, axis=1)
    idx_peri = int(np.argmin(dist_phase2))
    state_at_peri = traj_phase2[idx_peri]
    peri = find_perilune(dyn, state_at_peri, (0.0, 0.05), n_steps=2_000)

    print(f"  Perilune radius:      "
          f"{peri.r_perilune * LENGTH_KM:,.0f} km "
          f"(alt {peri.r_perilune * LENGTH_KM - MOON_RADIUS_KM:,.0f} km)")
    print(f"  Speed at perilune:    "
          f"{peri.v_perilune * VELOCITY_M_S:,.0f} m/s")
    print(f"  Local circular:       "
          f"{peri.v_circ * VELOCITY_M_S:,.0f} m/s")
    print(f"  Entry state:          "
          f"{'bound' if peri.is_bound else 'hyperbolic'}")

    # Sliding-window capture from the raw Phase-2 perilune state.
    def _phase3_sampler(qubo):
        if qubo.n_vars <= 18:
            return sample_brute_force(qubo).schedule
        return sample_simulated_annealing(
            qubo, num_reads=1500, seed=42,
        ).schedule

    # With high thrust we restrict the action region tightly to the
    # perilune passage (3x perilune radius) and use very short windows
    # (~1 hour) so the linearisation stays accurate. Outside the action
    # region the spacecraft drifts freely. With 3.5 mm/s^2 each 1-hour
    # window can deliver up to ~12 m/s of dV, so a few windows during
    # the close approach accumulate enough braking to bind the orbit.
    moon_action_radius = max(3.0 * peri.r_perilune, 0.04)
    capture = solve_lunar_capture_sliding_window(
        dyn, peri.state_synodic, sampler=_phase3_sampler,
        n_windows=200,
        window_revs=0.5,
        window_t_max=0.01,                   # ~ 1 hour per window
        moon_action_radius=moon_action_radius,
        drift_t_max=0.3,
        n_decision_steps=12,
        thrust_magnitude=ACCEL_MAIN_NONDIM,
        target_position_weight=0.0,
        target_velocity_weight=1.0,
        fuel_weight=0.0,
        brake_factor=1.0,
        target_radius=None,
        n_integration_substeps=80,
        n_truth_substeps=300,
        max_inner_iters=3,
        verbose=False,
    )

    dv_phase3_nondim = capture.total_delta_v
    e_init = moon_two_body_energy(peri.state_synodic)
    e_final = moon_two_body_energy(capture.final_state)
    apo_init, _, _ = moon_orbit_apolune_perilune(peri.state_synodic)
    apo_final, _, _ = moon_orbit_apolune_perilune(capture.final_state)

    print(f"  Windows:              {len(capture.windows)}")
    print(f"  Burns:                {capture.total_burns}")
    print(f"  dV_3:                 "
          f"{dv_phase3_nondim * VELOCITY_M_S:.1f} m/s")
    print(f"  Energy:               "
          f"{e_init:+.3e} -> {e_final:+.3e}")
    apo_init_km = apo_init * LENGTH_KM if np.isfinite(apo_init) else float('inf')
    apo_final_km = apo_final * LENGTH_KM if np.isfinite(apo_final) else float('inf')
    print(f"  Apolune:              "
          f"{apo_init_km:,.0f} km -> {apo_final_km:,.0f} km")
    print(f"  Captured:             "
          f"{'YES (bound to Moon)' if capture.captured else 'no'}")

    # -----------------------------------------------------------------
    # Total mission dV breakdown
    # -----------------------------------------------------------------
    _section("Total mission dV breakdown (all binary)")

    dv_total = dv_phase1_total_nondim + dv_phase2_nondim + dv_phase3_nondim
    rows = [
        ("Phase 1 (Edelbaum spiral + injection boost)",
         dv_phase1_total_nondim * VELOCITY_M_S),
        ("Phase 2 (cislunar QUBO correction)",
         dv_phase2_nondim * VELOCITY_M_S),
        ("Phase 3 (sliding-window QUBO capture)",
         dv_phase3_nondim * VELOCITY_M_S),
        ("TOTAL",
         dv_total * VELOCITY_M_S),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{width}}  {value:>10,.1f} m/s")

    print()
    # Phase-3 elapsed time: sum of per-window dt_decision * n_steps over
    # accepted windows (drift windows do not appear in capture.windows).
    tof_phase3_nondim = sum(
        w.iterative_result.final_qubo.dt_decision
        * w.iterative_result.final_qubo.n_steps
        for w in capture.windows
    )
    tof_total_nondim = (
        tof_phase1_total_nondim + T_PHASE2 + tof_phase3_nondim
    )
    tof_total_days = tof_total_nondim * TIME_S / 86_400.0
    tof_phase3_days = tof_phase3_nondim * TIME_S / 86_400.0
    print(f"  Phase-3 elapsed time: {tof_phase3_days:.1f} days "
          f"({len(capture.windows)} windows)")
    print(f"  Total time of flight: "
          f"~{tof_total_days:,.1f} days "
          f"({tof_total_days / 365.25:.3f} years)")
    print()
    print("  All thrust delivered by a single 350 mN ion/Hall thruster.\n"
          "  Every dV in the breakdown is a sequence of on/off binary\n"
          "  commands -- no impulsive burns.")


if __name__ == "__main__":
    main()
