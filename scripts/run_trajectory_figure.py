"""Publication-quality trajectory visualisation for the cislunar
correction scenario.

Two panels:
  (a) Synodic (rotating) frame: Earth, Moon, L1, the tail of the
      Phase-1 Edelbaum spiral (last ~12 revolutions), Phase-2 coast
      vs corrected trajectories. Burn intervals appear as thick red
      segments along the trajectory (showing duration, not points).
  (b) Earth-centered inertial frame: Phase-2 trajectory with the
      Moon's path and the corrected trajectory's burn segments.

Spacecraft characteristics for the scenario are reported in the
caption block (mass, thrust, on-time per slot).

Output: ``cislunar_trajectory.png`` next to this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.scheduling_samplers import sample_brute_force
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    propagate_schedule,
    solve_iterative,
)
from qalunar.reference.edelbaum import (
    ACCELERATION_M_S2,
    LENGTH_KM,
    TIME_S,
    VELOCITY_M_S,
    edelbaum_spiral,
)


_MU = EARTH_MOON_MU
_GM = 1.0 - _MU
_R_HEO = 200_000.0 / LENGTH_KM
_V_CIRC = float(np.sqrt(_GM / _R_HEO))
_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R = np.array([-_R_SYN[1], _R_SYN[0]])

# Scenario chosen empirically (see scripts/scenario_search) for a
# clear visual story: small under-injection that the QUBO partially
# corrects, putting the spacecraft visibly closer to the Moon.
V_RATIO = 1.188          # under-injection (perfect would be 1.190)
THRUST_MAG = 0.02        # nondim acceleration  ≈ 5.4 mm/s²
T_SPAN = (0.0, 4.0)      # nondim time          ≈ 17.4 days
N_STEPS = 20             # decision intervals; each Δt ≈ 21 hours

# Phase-1 Edelbaum spiral parameters (used only for the visual tail).
PHASE1_R1 = 6578.0 / LENGTH_KM             # 200-km LEO altitude
PHASE1_R2 = _R_HEO                          # 200,000-km HEO (= handoff)
PHASE1_A = 0.01                             # nondim accel for Phase-1
PHASE1_LAST_REVS = 12

SPACECRAFT_MASS_KG = 100.0


def _bf_sampler(qubo) -> np.ndarray:
    return sample_brute_force(qubo).schedule


def _spiral_synodic_tail(n_last_revs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_syn, y_syn) for the last ``n_last_revs`` revolutions of
    the Phase-1 Edelbaum spiral, expressed in CR3BP synodic coordinates
    and rotated so the spiral's last point coincides with the Phase-2
    handoff at synodic position ``(-μ + r2, 0)``.

    The Edelbaum analytical model places the spacecraft at radius r2 at
    the end of Phase-1 but at a phase angle ``θ_total mod 2π`` that is
    not generally aligned with the synodic +x axis.  Since the absolute
    Phase-1 phase is not constrained by the optimal-control problem, we
    rotate the trajectory by a constant offset so the visualisation
    connects cleanly to the QUBO scenario's handoff state.
    """
    sp = edelbaum_spiral(r1=PHASE1_R1, r2=PHASE1_R2, a_thrust=PHASE1_A)
    GM = 1.0 - sp.mu
    theta_total = (sp.v1**4 - sp.v2**4) / (4.0 * GM * sp.a_thrust)
    theta_min = theta_total - n_last_revs * 2.0 * np.pi
    if theta_min < 0:
        theta_min = 0.0
    v_at_min = (sp.v1**4 - 4.0 * GM * sp.a_thrust * theta_min) ** 0.25
    t_at_min = (sp.v1 - v_at_min) / sp.a_thrust

    n_pts = 12_000
    t = np.linspace(t_at_min, sp.tof, n_pts)
    v_t = sp.v1 - sp.a_thrust * t
    r_t = GM / v_t**2
    theta_t = (sp.v1**4 - v_t**4) / (4.0 * GM * sp.a_thrust)

    # Earth-centered inertial -> synodic (Earth at -μ in synodic).
    # The synodic angle is (θ - t).  At t = sp.tof this evaluates to
    # (θ_total - sp.tof) which is generally not zero.  We add a constant
    # angle offset so the spiral's last sample lands exactly on the
    # handoff state at (-μ + r2, 0).
    phi_at_end = theta_total - sp.tof
    phi = (theta_t - t) - phi_at_end          # synodic angle, with offset
    x_syn = r_t * np.cos(phi) + (-sp.mu)
    y_syn = r_t * np.sin(phi)
    return x_syn, y_syn


def synodic_to_eci(times: np.ndarray, xy_syn: np.ndarray,
                   anchor_syn: np.ndarray) -> np.ndarray:
    """Synodic (rotating) -> inertial frame, with origin at the body
    whose synodic position is ``anchor_syn``."""
    rel_syn = xy_syn - anchor_syn
    cos_t = np.cos(times)
    sin_t = np.sin(times)
    x = cos_t * rel_syn[:, 0] - sin_t * rel_syn[:, 1]
    y = sin_t * rel_syn[:, 0] + cos_t * rel_syn[:, 1]
    return np.column_stack([x, y])


def _burn_segment_indices(
    schedule: np.ndarray, dt_decision: float, sub_per_decision: int,
) -> list[tuple[int, int]]:
    """Map active decision slots to (start_idx, end_idx) ranges in the
    high-resolution propagated trajectory."""
    n_steps = schedule.size
    segments = []
    for i in range(n_steps):
        if schedule[i] == 1:
            i0 = i * sub_per_decision
            i1 = (i + 1) * sub_per_decision + 1  # inclusive endpoint
            segments.append((i0, i1))
    return segments


def main() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    dyn = PlanarCR3BP()
    state_perfect = np.concatenate([_R_SYN,
        np.array([0.0, _V_CIRC * 1.190]) - _OMEGA_CROSS_R])
    _, traj_perfect = dyn.propagate(state_perfect, T_SPAN, n_steps=8000)
    target = traj_perfect[-1]

    state0 = np.concatenate([_R_SYN,
        np.array([0.0, _V_CIRC * V_RATIO]) - _OMEGA_CROSS_R])

    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_direction="tangential",
        fuel_weight=0.0,
    )
    result = solve_iterative(
        dyn, state0, target, T_SPAN,
        n_decision_steps=N_STEPS, sampler=_bf_sampler, config=cfg,
        n_integration_substeps=80, n_truth_substeps=400,
        max_iters=8,
    )
    schedule = result.schedule
    miss_km = float(np.linalg.norm(result.true_miss)) * LENGTH_KM
    dv_m_s = result.final_qubo.delta_v_m_s(schedule)
    n_burns = int(schedule.sum())

    # Both trajectories at high resolution. propagate_schedule integrates
    # interval-by-interval at 400 substeps each.
    times_coast, traj_coast = dyn.propagate(state0, T_SPAN, n_steps=4000)
    sub_per_decision = 400
    times_corr, traj_corr = propagate_schedule(
        dyn, state0, T_SPAN, schedule, cfg,
        n_integration_substeps=sub_per_decision, return_trajectory=True,
    )
    burn_segments = _burn_segment_indices(
        schedule, dt_decision=(T_SPAN[1] - T_SPAN[0]) / N_STEPS,
        sub_per_decision=sub_per_decision,
    )

    # Spiral tail (synodic)
    x_sp, y_sp = _spiral_synodic_tail(PHASE1_LAST_REVS)

    # --- Build figure ---
    fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    ax_syn = fig.add_subplot(gs[0, 0])
    ax_eci = fig.add_subplot(gs[0, 1])

    earth_syn = np.array([-_MU, 0.0])
    moon_syn = np.array([1 - _MU, 0.0])
    L1 = np.array([1 - _MU - (_MU / 3) ** (1 / 3), 0.0])

    # ===========================================================
    # Panel (a): SYNODIC frame
    # ===========================================================
    ax = ax_syn

    # Phase 1 spiral tail (light, behind everything)
    ax.plot(x_sp, y_sp, "-", color="0.8", lw=0.6, alpha=0.85,
            label=f"Phase 1 spiral (last {PHASE1_LAST_REVS} revs)")

    # Coast Phase-2
    ax.plot(traj_coast[:, 0], traj_coast[:, 1], "--",
            color="0.45", lw=2.0, alpha=0.85,
            label="Phase 2: no correction")

    # Corrected Phase-2
    ax.plot(traj_corr[:, 0], traj_corr[:, 1], "-",
            color="tab:green", lw=2.4, alpha=0.95,
            label="Phase 2: QUBO-corrected")

    # Burn duration segments (thick red)
    for k, (i0, i1) in enumerate(burn_segments):
        seg = traj_corr[i0:i1]
        label = f"Burn intervals ({n_burns}×{(T_SPAN[1] - T_SPAN[0]) / N_STEPS * TIME_S / 3600:.0f} h)" if k == 0 else None
        ax.plot(seg[:, 0], seg[:, 1], "-",
                color="tab:red", lw=4.5, alpha=0.95, solid_capstyle="round",
                label=label, zorder=8)

    # Earth, Moon (drawn to scale)
    R_EARTH_NONDIM = 6378.0 / LENGTH_KM
    R_MOON_NONDIM = 1737.0 / LENGTH_KM
    ax.add_patch(Circle(earth_syn, R_EARTH_NONDIM * 4,  # exaggerated for visibility
                        color="tab:blue", alpha=0.85, zorder=5))
    ax.add_patch(Circle(moon_syn, R_MOON_NONDIM * 6,
                        color="0.3", alpha=0.85, zorder=5))
    ax.plot(*L1, "x", color="tab:purple", ms=11, mew=2.6, zorder=5)
    ax.annotate("Earth", earth_syn, textcoords="offset points",
                xytext=(8, 14), fontsize=12, fontweight="bold")
    ax.annotate("Moon", moon_syn, textcoords="offset points",
                xytext=(10, 14), fontsize=12, fontweight="bold")
    ax.annotate("$L_1$", (L1[0], L1[1]), textcoords="offset points",
                xytext=(-6, -22), fontsize=12, color="tab:purple")

    # Handoff marker
    ax.plot(state0[0], state0[1], "o",
            color="tab:green", ms=12, mec="black", mew=1.2, zorder=10)
    ax.annotate("Handoff\n(Phase 1 → 2)", (state0[0], state0[1]),
                textcoords="offset points", xytext=(-95, 10),
                fontsize=10, ha="left")

    # Smart axis limits: include spiral tail, both Phase-2 trajectories,
    # and the Moon, with ~5% padding.
    all_x = np.concatenate([x_sp, traj_coast[:, 0], traj_corr[:, 0],
                            np.array([earth_syn[0], moon_syn[0]])])
    all_y = np.concatenate([y_sp, traj_coast[:, 1], traj_corr[:, 1],
                            np.array([earth_syn[1], moon_syn[1]])])
    pad = 0.05 * (all_x.max() - all_x.min())
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

    ax.set_xlabel("$x$ (synodic, nondim)")
    ax.set_ylabel("$y$ (synodic, nondim)")
    ax.set_title("(a) Synodic (rotating) frame")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # ===========================================================
    # Panel (b): EARTH-CENTERED INERTIAL frame (Phase 2 only)
    # ===========================================================
    ax = ax_eci

    coast_eci = synodic_to_eci(times_coast, traj_coast[:, :2], earth_syn)
    corr_eci = synodic_to_eci(times_corr, traj_corr[:, :2], earth_syn)

    # Moon's path in ECI over the Phase-2 interval
    moon_t = np.linspace(T_SPAN[0], T_SPAN[1], 400)
    moon_eci = synodic_to_eci(moon_t,
                              np.tile(moon_syn, (len(moon_t), 1)),
                              earth_syn)

    # Coast and corrected
    ax.plot(coast_eci[:, 0], coast_eci[:, 1], "--",
            color="0.45", lw=2.0, alpha=0.85, label="Phase 2: no correction")
    ax.plot(corr_eci[:, 0], corr_eci[:, 1], "-",
            color="tab:green", lw=2.4, alpha=0.95, label="Phase 2: corrected")
    ax.plot(moon_eci[:, 0], moon_eci[:, 1], ":",
            color="0.5", lw=1.4, alpha=0.7, label="Moon's path")

    # Burn duration segments in ECI
    for k, (i0, i1) in enumerate(burn_segments):
        seg = corr_eci[i0:i1]
        label = "Burn intervals" if k == 0 else None
        ax.plot(seg[:, 0], seg[:, 1], "-",
                color="tab:red", lw=4.5, alpha=0.95, solid_capstyle="round",
                label=label, zorder=8)

    # Earth (origin), Moon at t_f
    ax.plot(0, 0, "o", color="tab:blue", ms=14, zorder=5)
    ax.annotate("Earth", (0, 0), textcoords="offset points",
                xytext=(8, 12), fontsize=12, fontweight="bold")
    moon_at_tf = moon_eci[-1]
    ax.plot(*moon_at_tf, "o", color="0.3", ms=11, zorder=5)
    ax.annotate("Moon @ $t_f$", moon_at_tf, textcoords="offset points",
                xytext=(8, 8), fontsize=11)
    ax.plot(*moon_eci[0], "o", color="0.3", ms=8, alpha=0.5, zorder=4)
    ax.annotate("Moon @ $t_0$", moon_eci[0], textcoords="offset points",
                xytext=(8, 8), fontsize=10, alpha=0.7)

    ax.set_xlabel("$x_\\mathrm{ECI}$ (nondim)")
    ax.set_ylabel("$y_\\mathrm{ECI}$ (nondim)")
    ax.set_title("(b) Earth-centered inertial frame (Phase 2)")
    # Place legend outside the data area on the upper-right so it never
    # collides with the trajectory tail.
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
              fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # ===========================================================
    # Spacecraft / mission characteristics caption block
    # ===========================================================
    dt_nondim = (T_SPAN[1] - T_SPAN[0]) / N_STEPS
    burn_hours = dt_nondim * TIME_S / 3600.0
    accel_mm_s2 = THRUST_MAG * ACCELERATION_M_S2 * 1000.0
    thrust_N = SPACECRAFT_MASS_KG * (accel_mm_s2 * 1e-3)
    dv_per_burn = THRUST_MAG * dt_nondim * VELOCITY_M_S

    # Closest-approach numbers for the caption
    dist_coast_km = np.linalg.norm(
        (traj_coast[:, :2] - moon_syn) * LENGTH_KM, axis=1)
    dist_corr_km = np.linalg.norm(
        (traj_corr[:, :2] - moon_syn) * LENGTH_KM, axis=1)
    ca_coast_km = float(dist_coast_km.min())
    ca_corr_km = float(dist_corr_km.min())

    box_text = (
        "SPACECRAFT  "
        f"mass {SPACECRAFT_MASS_KG:.0f} kg · "
        f"thrust {thrust_N*1000:.1f} mN · "
        f"a = {accel_mm_s2:.3f} mm/s$^{{2}}$\n"
        "MISSION    "
        f"Phase 2 duration {T_SPAN[1]:.1f} nondim "
        f"({T_SPAN[1] * TIME_S / 86400.0:.1f} d) · "
        f"slot {burn_hours:.0f} h · "
        f"$\\Delta v_{{\\mathrm{{slot}}}}$ = {dv_per_burn:.1f} m/s\n"
        "RESULT     "
        f"N = {N_STEPS} slots, {n_burns} burns · "
        f"$\\Delta v_{{\\mathrm{{total}}}}$ = {dv_m_s:.1f} m/s · "
        f"final-state miss {miss_km:,.0f} km · "
        f"Moon CA {ca_coast_km:,.0f} $\\to$ {ca_corr_km:,.0f} km"
    )
    fig.text(
        0.5, -0.02, box_text,
        fontsize=10, family="monospace",
        va="top", ha="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                  edgecolor="0.6", alpha=0.97),
    )

    out_path = Path(__file__).resolve().parent / "figures" / "cislunar_trajectory.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"  burns={n_burns}, dV={dv_m_s:.1f} m/s, miss={miss_km:.0f} km")
    print(f"  coast Moon CA: {ca_coast_km:,.0f} km")
    print(f"  corrected Moon CA: {ca_corr_km:,.0f} km")
    print(f"  spacecraft: {SPACECRAFT_MASS_KG:.0f} kg, "
          f"thrust {thrust_N*1000:.0f} mN, "
          f"a = {accel_mm_s2:.2f} mm/s²")
    print(f"  burn slot: {burn_hours:.1f} h, "
          f"dV/slot = {dv_per_burn:.1f} m/s")


if __name__ == "__main__":
    main()
