"""Moon-orbit station-keeping demonstration.

A complementary scenario to the cislunar injection-correction case.
The spacecraft starts on a lunar orbit at 5\\,000~km altitude that has
been perturbed by a small velocity error.  Without correction the
orbit drifts away (CR3BP perturbations and the velocity error
compound over multiple revolutions).  With the iterative QUBO
schedule the spacecraft maintains its orbit near the nominal.

This figure shows what the iterative QUBO does well: small,
linearisation-faithful corrections on top of an existing reference
trajectory.  The visual story is "satellite orbits the Moon":
several revolutions in the Moon-relative frame with burn intervals
shown along the trajectory.

Output: ``moon_orbit_stationkeeping.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.scheduling_samplers import (
    sample_brute_force,
    sample_simulated_annealing,
)
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
)


_MU = EARTH_MOON_MU
_GM_MOON = _MU
_MOON_X = 1.0 - _MU
_MOON_SYN = np.array([_MOON_X, 0.0])

# Lunar orbit nominal and perturbation
ORBIT_ALT_KM = 5_000.0          # altitude above Moon surface
R_ORBIT = (1737.0 + ORBIT_ALT_KM) / LENGTH_KM
PHI0_DEG = 90.0                  # initial phase angle around Moon (CCW from +x)
V_PERTURB = 0.005                # 0.5% tangential-velocity error
PERTURB_DIRECTION = "tangential" # "radial" or "tangential"

# Spacecraft / mission parameters
SPACECRAFT_MASS_KG = 100.0
THRUST_MAG = 0.02                # nondim ≈ 0.054 mm/s²

# Time / scheduling
N_REVS = 2                       # two lunar revolutions
T_REV_NONDIM = 2.0 * np.pi * np.sqrt(R_ORBIT**3 / _GM_MOON)
T_SPAN = (0.0, N_REVS * T_REV_NONDIM)
N_STEPS = 16                     # decision slots
DT_DECISION = (T_SPAN[1] - T_SPAN[0]) / N_STEPS


def _nominal_state(phase_rad: float, prograde: bool = True) -> np.ndarray:
    """Synodic-frame state on a circular Moon-relative orbit at given
    phase angle around the Moon.

    Constructs the spacecraft's *total* inertial velocity as the sum of
    the Moon's inertial velocity (= ``omega × r_moon``) and the
    Moon-relative orbital velocity, then converts to the synodic frame.
    """
    pos = _MOON_SYN + R_ORBIT * np.array([np.cos(phase_rad), np.sin(phase_rad)])
    v_circ = np.sqrt(_GM_MOON / R_ORBIT)
    sign = 1.0 if prograde else -1.0
    # Velocity of the spacecraft relative to the Moon (inertial frame)
    v_rel = sign * v_circ * np.array([-np.sin(phase_rad), np.cos(phase_rad)])
    # Moon's inertial velocity at t = 0 (frames aligned):
    # v_moon_inertial = omega × r_moon = (-y_moon, x_moon) = (0, 1-μ)
    v_moon_inertial = np.array([-_MOON_SYN[1], _MOON_SYN[0]])
    v_inertial = v_moon_inertial + v_rel
    # Synodic frame: v_syn = v_inertial − omega × r_spacecraft_from_barycenter
    omega_cross_r = np.array([-pos[1], pos[0]])
    v_syn = v_inertial - omega_cross_r
    return np.concatenate([pos, v_syn])


def _sampler(qubo) -> np.ndarray:
    if qubo.n_vars <= 18:
        return sample_brute_force(qubo).schedule
    return sample_simulated_annealing(qubo, num_reads=4000, seed=42).schedule


def _burn_segments(schedule: np.ndarray, sub_per: int) -> list[tuple[int, int]]:
    n = schedule.size
    out = []
    for i in range(n):
        if schedule[i] == 1:
            out.append((i * sub_per, (i + 1) * sub_per + 1))
    return out


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

    phi0 = np.deg2rad(PHI0_DEG)
    nominal0 = _nominal_state(phi0)

    # Apply a velocity perturbation to make the orbit drift.
    radial_hat = (nominal0[:2] - _MOON_SYN) / R_ORBIT
    tangential_hat = np.array([-radial_hat[1], radial_hat[0]])  # CCW tangent
    v_circ_nondim = float(np.sqrt(_GM_MOON / R_ORBIT))
    perturb_hat = tangential_hat if PERTURB_DIRECTION == "tangential" else radial_hat
    state0 = nominal0.copy()
    state0[2:] = state0[2:] + V_PERTURB * v_circ_nondim * perturb_hat

    # Target: where the *unperturbed nominal* actually is at t = T under
    # CR3BP dynamics. We propagate the nominal forward with high accuracy
    # and use its endpoint. This avoids the small phase mismatch between
    # the analytical Keplerian period and the true CR3BP period.
    _, traj_nom = dyn.propagate(nominal0, T_SPAN, n_steps=20_000)
    target = traj_nom[-1]
    nominal_full = traj_nom        # full nominal trajectory for plotting

    # Position-only weighting: a 1.5% tangential perturbation over two
    # revs leaves the spacecraft on essentially the same orbit but at
    # a different phase angle, so the L2 norm in full 4-state space is
    # dominated by the velocity-direction mismatch.  For a station-
    # keeping mission we care about position, not phase per se, so we
    # zero out the velocity components of the target weight.
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_direction="antitangential",
        fuel_weight=1e-7,                  # discourage redundant burns
        target_weights=np.array([1.0, 1.0, 0.0, 0.0]),
    )

    result = solve_iterative(
        dyn, state0, target, T_SPAN,
        n_decision_steps=N_STEPS, sampler=_sampler, config=cfg,
        n_integration_substeps=80, n_truth_substeps=300, max_iters=8,
    )
    schedule = result.schedule
    miss_km = float(np.linalg.norm(result.true_miss)) * LENGTH_KM
    dv_m_s = result.final_qubo.delta_v_m_s(schedule)
    n_burns = int(schedule.sum())
    pos_miss_km = float(np.linalg.norm(result.true_miss[:2])) * LENGTH_KM

    # High-resolution propagation for plotting
    sub_per = 300
    times_coast, traj_coast = dyn.propagate(state0, T_SPAN, n_steps=4000)
    coast_pos_miss_km = float(
        np.linalg.norm(traj_coast[-1, :2] - target[:2])) * LENGTH_KM
    times_corr, traj_corr = propagate_schedule(
        dyn, state0, T_SPAN, schedule, cfg,
        n_integration_substeps=sub_per, return_trajectory=True,
    )
    burn_seg = _burn_segments(schedule, sub_per)

    # Nominal orbit, actually propagated (to capture the genuine CR3BP
    # closed orbit, not the Keplerian-approximate circle).
    nominal_xy = nominal_full[:, :2]

    # Distances to Moon (km) for the side panel
    rel_coast_km = (traj_coast[:, :2] - _MOON_SYN) * LENGTH_KM
    rel_corr_km = (traj_corr[:, :2] - _MOON_SYN) * LENGTH_KM

    # --- Build figure ---
    fig = plt.figure(figsize=(15, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_syn = fig.add_subplot(gs[0, 0])
    ax_moon = fig.add_subplot(gs[0, 1])

    # ===========================================================
    # Panel (a): SYNODIC FRAME (zoomed near Moon)
    # ===========================================================
    ax = ax_syn

    # Nominal orbit (target)
    ax.plot(nominal_xy[:, 0], nominal_xy[:, 1], ":",
            color="0.5", lw=1.4, alpha=0.8,
            label=f"Nominal {ORBIT_ALT_KM:.0f}-km orbit")

    # Coast (drift away)
    ax.plot(traj_coast[:, 0], traj_coast[:, 1], "--",
            color="0.45", lw=2.0, alpha=0.85,
            label="No correction (drift)")

    # Corrected
    ax.plot(traj_corr[:, 0], traj_corr[:, 1], "-",
            color="tab:green", lw=2.2, alpha=0.95,
            label="QUBO-corrected")

    # Burn segments
    for k, (i0, i1) in enumerate(burn_seg):
        seg = traj_corr[i0:i1]
        label = f"Burns ({n_burns} active)" if k == 0 else None
        ax.plot(seg[:, 0], seg[:, 1], "-",
                color="tab:red", lw=4.5, alpha=0.95, solid_capstyle="round",
                label=label, zorder=8)

    # Moon body to scale
    R_MOON = 1737.0 / LENGTH_KM
    ax.add_patch(Circle(_MOON_SYN, R_MOON, color="0.3", alpha=0.85, zorder=5))
    ax.annotate("Moon", _MOON_SYN,
                textcoords="offset points", xytext=(8, 14),
                fontsize=12, fontweight="bold")

    # Initial state
    ax.plot(state0[0], state0[1], "o",
            color="tab:green", ms=12, mec="black", mew=1.2, zorder=10)
    ax.annotate("Start (perturbed)", (state0[0], state0[1]),
                textcoords="offset points", xytext=(12, 6), fontsize=11)

    # Auto-zoom around Moon with margin
    half = 1.6 * R_ORBIT
    ax.set_xlim(_MOON_X - half, _MOON_X + half)
    ax.set_ylim(-half, half)
    ax.set_xlabel("$x$ (synodic, nondim)")
    ax.set_ylabel("$y$ (synodic, nondim)")
    ax.set_title("(a) Synodic frame — Moon zoom")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # ===========================================================
    # Panel (b): MOON-RELATIVE FRAME
    # ===========================================================
    ax = ax_moon
    nominal_rel_km = (nominal_full[:, :2] - _MOON_SYN) * LENGTH_KM
    ax.plot(nominal_rel_km[:, 0], nominal_rel_km[:, 1],
            ":", color="0.5", lw=1.4, alpha=0.8,
            label=f"Nominal {ORBIT_ALT_KM:.0f}-km orbit")
    ax.plot(rel_coast_km[:, 0], rel_coast_km[:, 1], "--",
            color="0.45", lw=2.0, alpha=0.85, label="No correction")
    ax.plot(rel_corr_km[:, 0], rel_corr_km[:, 1], "-",
            color="tab:green", lw=2.2, alpha=0.95, label="QUBO-corrected")
    for k, (i0, i1) in enumerate(burn_seg):
        seg_km = rel_corr_km[i0:i1]
        label = "Burns" if k == 0 else None
        ax.plot(seg_km[:, 0], seg_km[:, 1], "-",
                color="tab:red", lw=4.5, alpha=0.95, solid_capstyle="round",
                label=label, zorder=8)

    ax.add_patch(Circle((0, 0), 1737.0, color="0.3", alpha=0.85, zorder=5))
    ax.annotate("Moon", (0, 0),
                textcoords="offset points", xytext=(8, 12),
                fontsize=12, fontweight="bold")
    ax.plot(rel_corr_km[0, 0], rel_corr_km[0, 1], "o",
            color="tab:green", ms=12, mec="black", mew=1.2, zorder=10)

    half_km = 1.6 * R_ORBIT * LENGTH_KM
    ax.set_xlim(-half_km, half_km)
    ax.set_ylim(-half_km, half_km)
    ax.set_xlabel("Moon-relative $x$ (km)")
    ax.set_ylabel("Moon-relative $y$ (km)")
    ax.set_title(f"(b) Moon-relative frame ({N_REVS} revs over "
                 f"{T_SPAN[1] * TIME_S / 86400.0:.1f} d)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # Bottom caption block
    accel_mm_s2 = THRUST_MAG * ACCELERATION_M_S2 * 1000.0
    thrust_N = SPACECRAFT_MASS_KG * (accel_mm_s2 * 1e-3)
    burn_hours = DT_DECISION * TIME_S / 3600.0
    dv_per_burn = THRUST_MAG * DT_DECISION * VELOCITY_M_S

    n_channels_actual = result.schedule.size // N_STEPS
    box_text = (
        "SPACECRAFT  "
        f"mass {SPACECRAFT_MASS_KG:.0f} kg · "
        f"thrust {thrust_N*1000:.1f} mN · "
        f"a = {accel_mm_s2:.3f} mm/s$^{{2}}$ · "
        f"single anti-tangential channel\n"
        "MISSION    "
        f"{ORBIT_ALT_KM:.0f}-km lunar orbit · "
        f"{V_PERTURB*100:.1f}% {PERTURB_DIRECTION} velocity perturbation at $t_0$ · "
        f"horizon {T_SPAN[1]:.3f} nondim "
        f"({T_SPAN[1] * TIME_S / 86400.0:.1f} d, {N_REVS} revs)\n"
        "RESULT     "
        f"N = {N_STEPS} slots × {n_channels_actual} ch = "
        f"{result.schedule.size} qubits · "
        f"{n_burns} burns · "
        f"$\\Delta v_{{\\mathrm{{total}}}}$ = {dv_m_s:.1f} m/s · "
        f"position miss: coast {coast_pos_miss_km:,.0f} km "
        f"$\\to$ corrected {pos_miss_km:,.0f} km"
    )
    fig.text(
        0.5, -0.02, box_text,
        fontsize=10, family="monospace",
        va="top", ha="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                  edgecolor="0.6", alpha=0.97),
    )

    out_path = Path(__file__).resolve().parent / "figures" / "moon_orbit_stationkeeping.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"  burns={n_burns}/{result.schedule.size}, dV={dv_m_s:.1f} m/s")
    print(f"  position miss: coast {coast_pos_miss_km:,.0f} km "
          f"-> corrected {pos_miss_km:,.0f} km")
    print(f"  full state miss (4D): {miss_km:,.0f} km")
    print(f"  spacecraft: {SPACECRAFT_MASS_KG:.0f} kg, "
          f"thrust {thrust_N*1000:.1f} mN, a = {accel_mm_s2:.3f} mm/s²")
    print(f"  burn slot {burn_hours:.1f} h, dV/slot {dv_per_burn:.1f} m/s")


if __name__ == "__main__":
    main()
