"""End-to-end mission visualisation: Phases 1 - 3 in a single figure.

Companion plot to ``scripts/run_full_mission_demo.py``. Produces a
3-panel hero figure that shows every trajectory of a low-thrust
Earth-Moon mission:

    (a) Phase 1: Edelbaum Earth-ascending spiral (Earth-centered).
    (b) Phase 2: cislunar transfer with QUBO-corrected burns
        (synodic frame, Earth-Moon system).
    (c) Phase 3: sliding-window QUBO lunar capture (Moon-centered).

The Δv breakdown and TOF totals are reported in the accompanying
text and table rather than as a fourth bar-chart panel.

The mission scales differ by orders of magnitude (Earth-centered orbits
of tens of thousands of km; cislunar arc of ~400,000 km; lunar orbits
of thousands of km), so each phase is plotted in its natural frame.

Output: ``mission_overview.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.lunar_capture import (
    moon_orbit_apolune_perilune,
    moon_two_body_energy,
    reconstruct_full_trajectory,
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
    capture_delta_v,
    edelbaum_spiral,
    find_perilune,
    moon_relative_inertial,
)
from qalunar.reference.edelbaum import (
    ACCELERATION_M_S2,
    LENGTH_KM,
    TIME_S,
    VELOCITY_M_S,
)


# ---------------------------------------------------------------------------
# Mission parameters (matched to run_full_mission_demo.py)
# ---------------------------------------------------------------------------

SPACECRAFT_MASS_KG = 100.0
THRUST_MAIN_MN = 350.0
THRUST_CRUISE_MN = 5.4
ACCEL_MAIN_NONDIM = 1.286     # 3.5 mm/s^2
ACCEL_CRUISE_NONDIM = 0.02    # 0.054 mm/s^2

EARTH_RADIUS_KM = 6_378.0
MOON_RADIUS_KM = 1_737.0
R_GEO_KM = 35_786.0

R_PHASE1_START_KM = R_GEO_KM
R_HEO_KM = 200_000.0
A_PHASE1 = ACCEL_MAIN_NONDIM

V_RATIO_PERFECT = 1.190
V_RATIO_INJECTION = 1.187
T_PHASE2 = 4.0
N_PHASE2 = 15

LLO_ALT_KM = 100.0

_MU = EARTH_MOON_MU
_MOON_POS = np.array([1.0 - _MU, 0.0])
_EARTH_POS = np.array([-_MU, 0.0])


# ---------------------------------------------------------------------------
# Spiral point generators (Earth-centered ascending, Moon-centered descending)
# ---------------------------------------------------------------------------


def _spiral_xyt(
    spiral, gm: float, n_points: int = 60_000,
    descending: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the FULL spiral in body-relative km, with normalised
    progression t in [0, 1] for use with a colormap.

    The spiral is sampled at uniform time, so the colour gradient maps
    to time-of-flight.
    """
    t = np.linspace(0.0, spiral.tof, n_points)
    sign = +1.0 if descending else np.sign(spiral.v2 - spiral.v1)
    v_t = spiral.v1 + sign * spiral.a_thrust * t
    r_t = gm / v_t ** 2
    dt = t[1] - t[0]
    omega_inst = v_t ** 3 / gm
    theta = np.cumsum(omega_inst) * dt

    x = r_t * np.cos(theta) * LENGTH_KM
    y = r_t * np.sin(theta) * LENGTH_KM
    progress = t / t[-1]
    return x, y, progress


def _plot_spiral(
    ax, x: np.ndarray, y: np.ndarray, progress: np.ndarray,
    cmap_name: str, n_subsample: int = 6_000, lw: float = 0.7,
    alpha: float = 0.9,
) -> None:
    """Plot a spiral with a time-progression colour gradient.

    Uses ``matplotlib.collections.LineCollection`` so the spiral is
    rendered as a single artist coloured by progression along the path
    (early in light shade, late in dark shade).
    """
    if x.size > n_subsample:
        idx = np.linspace(0, x.size - 1, n_subsample).astype(int)
        x, y, progress = x[idx], y[idx], progress[idx]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments, cmap=plt.get_cmap(cmap_name),
        norm=plt.Normalize(0.0, 1.0), linewidth=lw, alpha=alpha,
    )
    lc.set_array(progress[:-1])
    ax.add_collection(lc)


# ---------------------------------------------------------------------------
# Mission run (computes everything; plotting follows)
# ---------------------------------------------------------------------------


def run_mission() -> dict:
    dyn = PlanarCR3BP()

    # ---- Phase 1: Edelbaum spiral + injection boost -----------------
    r_start = (EARTH_RADIUS_KM + R_PHASE1_START_KM) / LENGTH_KM
    r_heo = R_HEO_KM / LENGTH_KM
    spiral1 = edelbaum_spiral(r1=r_start, r2=r_heo, a_thrust=A_PHASE1)
    r_handoff_1, _ = spiral1.handoff_state_synodic()

    omega_cross_r1 = np.array([-r_handoff_1[1], r_handoff_1[0]])
    v_circ_local = float(np.sqrt((1.0 - _MU) / r_heo))
    dv_injection_nondim = float(
        v_circ_local * (V_RATIO_INJECTION - 1.0)
    )
    tof_inject_nondim = dv_injection_nondim / ACCEL_MAIN_NONDIM
    dv_phase1_total_nondim = spiral1.dv + dv_injection_nondim
    tof_phase1_total_nondim = spiral1.tof + tof_inject_nondim

    state_perfect = np.concatenate([
        r_handoff_1,
        np.array([0.0, v_circ_local * V_RATIO_PERFECT]) - omega_cross_r1,
    ])
    state_after_phase1 = np.concatenate([
        r_handoff_1,
        np.array([0.0, v_circ_local * V_RATIO_INJECTION]) - omega_cross_r1,
    ])

    # ---- Phase 2: cislunar QUBO at CRUISE thrust -------------------
    _, traj_perfect = dyn.propagate(state_perfect, (0.0, T_PHASE2),
                                    n_steps=8_000)
    target_phase2 = traj_perfect[-1]

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

    # Reconstruct corrected Phase-2 trajectory (high resolution for plotting)
    sub_per = 300
    times_p2, traj_p2 = propagate_schedule(
        dyn, state_after_phase1, (0.0, T_PHASE2), schedule, cfg,
        n_integration_substeps=sub_per, return_trajectory=True,
    )
    burn_indices: list[tuple[int, int]] = []
    for i in range(N_PHASE2):
        if schedule[i] == 1:
            burn_indices.append((i * sub_per, (i + 1) * sub_per + 1))

    # Coast (uncorrected) for comparison
    _, traj_coast = dyn.propagate(state_after_phase1, (0.0, T_PHASE2),
                                  n_steps=4_000)

    # Perilune detection
    dist_p2 = np.linalg.norm(traj_p2[:, :2] - _MOON_POS, axis=1)
    idx_peri = int(np.argmin(dist_p2))
    state_at_peri = traj_p2[idx_peri]
    peri = find_perilune(dyn, state_at_peri, (0.0, 0.05), n_steps=2_000)

    # ---- Phase 3: sliding-window QUBO at MAIN thrust ----------------
    # No pre-capture burn: with 350 mN main thrust each 1-hour window
    # delivers up to ~12 m/s of dV. A few windows during the perilune
    # close approach accumulate enough braking to drop the orbit below
    # escape velocity. The action-radius is restricted to ~3 perilune
    # radii so the QUBO only fires near the Moon (where anti-tangential
    # thrust effectively dissipates orbital energy).
    def _sampler(qubo):
        if qubo.n_vars <= 18:
            return sample_brute_force(qubo).schedule
        return sample_simulated_annealing(
            qubo, num_reads=1500, seed=42,
        ).schedule

    moon_action_radius = max(3.0 * peri.r_perilune, 0.04)
    capture_result = solve_lunar_capture_sliding_window(
        dyn, peri.state_synodic, sampler=_sampler,
        n_windows=200,
        window_revs=0.5,
        window_t_max=0.01,
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

    times_p3, traj_p3 = reconstruct_full_trajectory(
        dyn, capture_result, thrust_magnitude=ACCEL_MAIN_NONDIM,
        n_integration_substeps=200,
    )

    return {
        "spiral1": spiral1,
        "dv_injection_nondim": dv_injection_nondim,
        "tof_inject_nondim": tof_inject_nondim,
        "dv_phase1_total_nondim": dv_phase1_total_nondim,
        "tof_phase1_total_nondim": tof_phase1_total_nondim,
        "result_phase2": result,
        "schedule": schedule,
        "traj_p2": traj_p2,
        "traj_coast": traj_coast,
        "burn_indices": burn_indices,
        "idx_peri": idx_peri,
        "perilune": peri,
        "capture_result": capture_result,
        "traj_p3": traj_p3,
        "times_p3": times_p3,
        "state_after_phase1": state_after_phase1,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_mission(data: dict, out_path: Path) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(18, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])
    ax_p1 = fig.add_subplot(gs[0, 0])
    ax_p2 = fig.add_subplot(gs[0, 1])
    ax_p3 = fig.add_subplot(gs[0, 2])

    # =====================================================
    # (a) Phase 1: Earth-centered ascending spiral
    # =====================================================
    spiral1 = data["spiral1"]
    x_p1, y_p1, prog_p1 = _spiral_xyt(spiral1, gm=1.0 - _MU,
                                      descending=False)
    _plot_spiral(ax_p1, x_p1, y_p1, prog_p1, cmap_name="Blues",
                 lw=0.7, alpha=0.85)

    # Start (light) and end (dark) markers on the spiral
    ax_p1.plot(x_p1[0], y_p1[0], "o",
               color="tab:cyan", ms=8, mec="black", mew=0.8, zorder=10)
    ax_p1.plot(x_p1[-1], y_p1[-1], "o",
               color="navy", ms=8, mec="black", mew=0.8, zorder=10)

    # Earth body to scale
    ax_p1.add_patch(Circle((0, 0), EARTH_RADIUS_KM,
                           color="tab:blue", alpha=0.6, zorder=5))
    ax_p1.plot(0, 0, "o", color="tab:blue", ms=6, zorder=6)
    ax_p1.annotate("Earth", (0, 0), textcoords="offset points",
                   xytext=(8, 8), fontsize=10, fontweight="bold")

    # Start and end reference orbits
    r_start_km = (EARTH_RADIUS_KM + R_PHASE1_START_KM)
    ax_p1.add_patch(Circle((0, 0), r_start_km, fill=False,
                           color="tab:cyan", ls="--", lw=1.0, alpha=0.8,
                           label=f"Start (GEO, {R_PHASE1_START_KM:,.0f} km)"))
    r_heo_km = R_HEO_KM
    ax_p1.add_patch(Circle((0, 0), r_heo_km, fill=False,
                           color="navy", ls="--", lw=1.0, alpha=0.8,
                           label=f"End (parking, {R_HEO_KM:,.0f} km)"))

    span_km = 1.15 * r_heo_km
    ax_p1.set_xlim(-span_km, span_km)
    ax_p1.set_ylim(-span_km, span_km)
    ax_p1.set_xlabel("Earth-centered x (km)")
    ax_p1.set_ylabel("Earth-centered y (km)")
    ax_p1.set_title(
        f"(a) Phase 1: Edelbaum spiral, "
        f"GEO -> {R_HEO_KM/1000:.0f}k km parking"
    )
    ax_p1.set_aspect("equal")
    ax_p1.grid(alpha=0.3)
    ax_p1.legend(loc="upper right", fontsize=8)

    metrics_p1 = (
        f"dV: {spiral1.dv * VELOCITY_M_S:,.0f} m/s\n"
        f"TOF: {spiral1.tof * TIME_S / 86_400.0:,.0f} days\n"
        f"Revs: {spiral1.n_revs:,.0f}"
    )
    ax_p1.text(0.02, 0.98, metrics_p1, transform=ax_p1.transAxes,
               fontsize=8.5, family="monospace", va="top", ha="left",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                         edgecolor="0.6", alpha=0.95))

    # =====================================================
    # (b) Phase 2: Synodic Earth-Moon system
    # =====================================================
    traj_p2 = data["traj_p2"]
    traj_coast = data["traj_coast"]
    burn_indices = data["burn_indices"]
    peri = data["perilune"]
    idx_peri = data["idx_peri"]
    schedule = data["schedule"]
    n_burns = int(schedule.sum())

    # Truncate Phase 2 at perilune so it visually connects with the
    # Phase 3 capture trajectory (which begins from that same point).
    traj_p2_to_peri = traj_p2[: idx_peri + 1]

    # Coast (drift)
    ax_p2.plot(traj_coast[:, 0], traj_coast[:, 1], "--",
               color="0.55", lw=1.2, alpha=0.7, label="Coast (no correction)")

    # Corrected trajectory (truncated at perilune)
    ax_p2.plot(traj_p2_to_peri[:, 0], traj_p2_to_peri[:, 1], "-",
               color="tab:green", lw=1.5, alpha=0.95,
               label="QUBO-corrected (to perilune)")

    # Burn segments — only those that occur before perilune
    burns_drawn = 0
    for i0, i1 in burn_indices:
        if i0 > idx_peri:
            continue
        i1_clipped = min(i1, idx_peri + 1)
        seg = traj_p2[i0:i1_clipped]
        label = (f"Burns ({n_burns} active)"
                 if burns_drawn == 0 else None)
        ax_p2.plot(seg[:, 0], seg[:, 1], "-",
                   color="tab:red", lw=3.0, alpha=0.95, label=label,
                   solid_capstyle="round", zorder=8)
        burns_drawn += 1

    # Earth and Moon
    ax_p2.plot(*_EARTH_POS, "o", color="tab:blue", ms=11, zorder=5)
    ax_p2.annotate("Earth", _EARTH_POS, textcoords="offset points",
                   xytext=(-8, 10), fontsize=10, fontweight="bold")
    ax_p2.plot(*_MOON_POS, "o", color="dimgray", ms=7, zorder=5)
    ax_p2.annotate("Moon", _MOON_POS, textcoords="offset points",
                   xytext=(8, 10), fontsize=10, fontweight="bold")

    # Perilune marker
    peri_xy = peri.state_synodic[:2]
    ax_p2.plot(peri_xy[0], peri_xy[1], "*",
               color="tab:purple", ms=18, mec="black", mew=0.8, zorder=10,
               label=f"Perilune ({peri.r_perilune * LENGTH_KM:,.0f} km)")

    # Start
    ax_p2.plot(traj_p2[0, 0], traj_p2[0, 1], "o",
               color="tab:green", ms=10, mec="black", mew=0.8, zorder=10)
    ax_p2.annotate("Start", (traj_p2[0, 0], traj_p2[0, 1]),
                   textcoords="offset points", xytext=(10, -10), fontsize=9)

    ax_p2.set_xlim(-0.15, 1.35)
    ax_p2.set_ylim(-0.4, 1.0)
    ax_p2.set_xlabel("x (synodic, nondim)")
    ax_p2.set_ylabel("y (synodic, nondim)")
    ax_p2.set_title("(b) Phase 2: cislunar transfer with QUBO correction")
    ax_p2.set_aspect("equal")
    ax_p2.grid(alpha=0.3)
    ax_p2.legend(loc="lower left", fontsize=8)

    dv_phase2_m_s = (
        data["result_phase2"].final_qubo.delta_v(schedule) * VELOCITY_M_S
    )
    miss_km = float(
        np.linalg.norm(data["result_phase2"].true_miss[:2])
    ) * LENGTH_KM
    metrics_p2 = (
        f"dV: {dv_phase2_m_s:,.1f} m/s\n"
        f"TOF: {T_PHASE2 * TIME_S / 86_400.0:.1f} days\n"
        f"Burns: {n_burns}/{N_PHASE2}\n"
        f"Iters: {data['result_phase2'].iterations}\n"
        f"Miss: {miss_km:,.0f} km"
    )
    ax_p2.text(0.02, 0.98, metrics_p2, transform=ax_p2.transAxes,
               fontsize=8.5, family="monospace", va="top", ha="left",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                         edgecolor="0.6", alpha=0.95))

    # =====================================================
    # (c) Phase 3: sliding-window QUBO capture trajectory
    # =====================================================
    capture_result = data["capture_result"]
    traj_p3 = data["traj_p3"]
    rel_xy_km = (traj_p3[:, :2] - _MOON_POS) * LENGTH_KM
    progress_p3 = np.linspace(0.0, 1.0, len(rel_xy_km))

    # Trajectory with time-progression colormap
    points = np.array([rel_xy_km[:, 0], rel_xy_km[:, 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments, cmap=plt.get_cmap("Greens"),
        norm=plt.Normalize(0.0, 1.0), linewidth=0.7, alpha=0.9,
    )
    lc.set_array(progress_p3[:-1])
    ax_p3.add_collection(lc)

    # Moon body to scale
    ax_p3.add_patch(Circle((0, 0), MOON_RADIUS_KM,
                           color="dimgray", alpha=0.7, zorder=5))
    ax_p3.plot(0, 0, "o", color="dimgray", ms=4, zorder=6)
    ax_p3.annotate("Moon", (0, 0), textcoords="offset points",
                   xytext=(8, 8), fontsize=10, fontweight="bold")

    # Start and end markers
    ax_p3.plot(rel_xy_km[0, 0], rel_xy_km[0, 1], "o",
               color="lime", ms=10, mec="black", mew=0.8, zorder=10,
               label="Start (after pre-capture)")
    ax_p3.plot(rel_xy_km[-1, 0], rel_xy_km[-1, 1], "*",
               color="darkgreen", ms=14, mec="black", mew=0.8, zorder=10,
               label="End (stabilised orbit)")

    span_km = 1.05 * np.max(np.abs(rel_xy_km))
    ax_p3.set_xlim(-span_km, span_km)
    ax_p3.set_ylim(-span_km, span_km)
    ax_p3.set_xlabel("Moon-centered x (km)")
    ax_p3.set_ylabel("Moon-centered y (km)")
    ax_p3.set_title(
        "(c) Phase 3: sliding-window QUBO lunar capture (binary thrust)"
    )
    ax_p3.set_aspect("equal")
    ax_p3.grid(alpha=0.3)
    ax_p3.legend(loc="upper right", fontsize=8)

    e0_p3 = moon_two_body_energy(peri.state_synodic)
    e_final_p3 = moon_two_body_energy(capture_result.final_state)
    apo0_p3, _, _ = moon_orbit_apolune_perilune(peri.state_synodic)
    apo_final_p3, _, _ = moon_orbit_apolune_perilune(
        capture_result.final_state
    )
    apo0_str = (f"{apo0_p3 * LENGTH_KM:,.0f}"
                if np.isfinite(apo0_p3) else "inf")
    apo_final_str = (f"{apo_final_p3 * LENGTH_KM:,.0f}"
                     if np.isfinite(apo_final_p3) else "inf")

    metrics_p3 = (
        f"main thrust:    {THRUST_MAIN_MN:.0f} mN\n"
        f"QUBO dV:        {capture_result.total_delta_v * VELOCITY_M_S:.0f} m/s\n"
        f"windows:        {len(capture_result.windows)}\n"
        f"burns:          {capture_result.total_burns}\n"
        f"e_init:         {e0_p3:+.3e}\n"
        f"e_final:        {e_final_p3:+.3e}\n"
        f"r_apo: {apo0_str} -> {apo_final_str} km\n"
        f"captured:       {'YES' if capture_result.captured else 'no'}"
    )
    ax_p3.text(0.02, 0.98, metrics_p3, transform=ax_p3.transAxes,
               fontsize=8.0, family="monospace", va="top", ha="left",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                         edgecolor="0.6", alpha=0.95))

    # Top-level title and footer
    fig.suptitle(
        f"All-binary Earth-Moon mission: "
        f"{SPACECRAFT_MASS_KG:.0f} kg microsatellite, "
        f"throttleable ion/Hall thruster "
        f"({THRUST_MAIN_MN:.0f} mN main / {THRUST_CRUISE_MN:.1f} mN cruise)",
        fontsize=12, fontweight="bold",
    )

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Computing all-binary mission...")
    data = run_mission()
    print("  Phase 1 dV (spiral + boost):",
          f"{data['dv_phase1_total_nondim'] * VELOCITY_M_S:,.0f} m/s")
    print("  Phase 2 dV:",
          f"{data['result_phase2'].final_qubo.delta_v(data['schedule']) * VELOCITY_M_S:,.1f} m/s")
    print("  Phase 3 dV (sliding-window QUBO):",
          f"{data['capture_result'].total_delta_v * VELOCITY_M_S:.1f} m/s")
    print(f"  Phase 3 captured: "
          f"{data['capture_result'].captured}")

    out_path = Path(__file__).resolve().parent / "figures" / "mission_overview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Plotting to {out_path} ...")
    plot_mission(data, out_path)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
