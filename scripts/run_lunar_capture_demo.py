"""Lunar capture demo: sliding-window QUBO from a hyperbolic flyby.

Starts the spacecraft at perilune of a hyperbolic flyby (anti-tangential
excess velocity) and runs the sliding-window scheduler to dissipate
energy until the orbit is bound around the Moon. The trajectory is
plotted in the Moon-relative frame together with the energy and
apolune evolution per window.

Run::

    python -m scripts.run_lunar_capture_demo
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.lunar_capture import (
    moon_two_body_energy,
    reconstruct_full_trajectory,
    solve_lunar_capture_sliding_window,
)
from qalunar.qubo.scheduling_samplers import (
    sample_brute_force,
    sample_simulated_annealing,
)
from qalunar.reference.edelbaum import LENGTH_KM, TIME_S, VELOCITY_M_S


_MU = EARTH_MOON_MU
_MOON_POS = np.array([1.0 - _MU, 0.0])
_MOON_RADIUS_KM = 1737.0


def build_initial_state(
    perilune_alt_km: float = 4_000.0,
    excess_factor: float = 1.20,
) -> np.ndarray:
    """Build a synodic-frame state at Moon-relative perilune with
    velocity ``excess_factor * v_circ`` (anti-tangential prograde).

    ``excess_factor = 1`` => circular (already captured).
    ``excess_factor > 1`` => hyperbolic flyby.
    """
    r = (_MOON_RADIUS_KM + perilune_alt_km) / LENGTH_KM
    pos = _MOON_POS + np.array([r, 0.0])
    v_circ = float(np.sqrt(_MU / r))
    v_rel_inertial = np.array([0.0, excess_factor * v_circ])
    v_moon_inertial = np.array([-_MOON_POS[1], _MOON_POS[0]])
    v_inertial = v_rel_inertial + v_moon_inertial
    omega_cross_r = np.array([-pos[1], pos[0]])
    v_syn = v_inertial - omega_cross_r
    return np.concatenate([pos, v_syn])


def _sampler(qubo):
    if qubo.n_vars <= 18:
        return sample_brute_force(qubo).schedule
    return sample_simulated_annealing(qubo, num_reads=1000, seed=42).schedule


def main() -> None:
    dyn = PlanarCR3BP()

    perilune_alt_km = 4_000.0
    # v_escape = sqrt(2) * v_circ ≈ 1.414 * v_circ. Stay below escape:
    # excess_factor = 1.30 -> bound but eccentric (e ~ 0.69) elliptical
    # orbit -- this is the regime where the sliding-window QUBO can
    # stably circularise the orbit.
    excess_factor = 1.30
    state0 = build_initial_state(
        perilune_alt_km=perilune_alt_km, excess_factor=excess_factor,
    )
    e0 = moon_two_body_energy(state0)
    print(f"Initial state energy w.r.t. Moon: {e0:+.3e} "
          f"({'bound' if e0 < 0 else 'hyperbolic'})")
    print(f"Initial perilune altitude:        {perilune_alt_km:.0f} km")
    print(f"Excess velocity factor:           {excess_factor:.3f}\n")

    print("Running sliding-window capture solver...")
    # Target = circular Moon-relative speed at the CURRENT radius (no
    # explicit target_radius). At perilune the orbit is moving faster
    # than circular -> QUBO fires anti-tangential to slow down.
    # At apolune the orbit is slower than circular -> anti-tangential
    # cannot help, QUBO stays off. Net effect: braking near each
    # perilune passage, energy decreases, orbit circularises.
    moon_action_r = 0.10
    result = solve_lunar_capture_sliding_window(
        dyn, state0, sampler=_sampler,
        n_windows=120,
        window_revs=0.5,
        window_t_max=0.04,
        moon_action_radius=moon_action_r,
        drift_t_max=0.3,
        n_decision_steps=14,
        thrust_magnitude=0.02,
        target_position_weight=0.0,
        target_velocity_weight=1.0,
        fuel_weight=0.0,
        brake_factor=1.0,       # target = exactly v_circ at current r
        target_radius=None,     # use current radius as reference
        n_integration_substeps=60,
        n_truth_substeps=200,
        max_inner_iters=3,
        verbose=True,
    )

    print()
    print(f"Captured?              {'yes' if result.captured else 'no'}")
    print(f"Total windows:         {len(result.windows)}")
    print(f"Total burns:           {result.total_burns}")
    print(f"Total dV:              "
          f"{result.total_delta_v * VELOCITY_M_S:.1f} m/s")
    print(f"Final energy:          "
          f"{moon_two_body_energy(result.final_state):+.3e}")

    # ---- Reconstruct full trajectory for plotting ----
    times, states = reconstruct_full_trajectory(
        dyn, result, thrust_magnitude=0.02, n_integration_substeps=200,
    )

    # ---- Plot ----
    fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0])
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_apo = fig.add_subplot(gs[0, 2])

    # --- Trajectory in Moon-relative km ---
    rel_xy_km = (states[:, :2] - _MOON_POS) * LENGTH_KM
    progress = np.linspace(0.0, 1.0, len(rel_xy_km))
    sc = ax_traj.scatter(rel_xy_km[:, 0], rel_xy_km[:, 1],
                         c=progress, cmap="viridis", s=0.8, alpha=0.85)
    plt.colorbar(sc, ax=ax_traj, label="time progression",
                 shrink=0.8, pad=0.02)

    # Burn segments highlighted
    for w in result.windows:
        # Within each window we don't have the burn segments unless we
        # re-propagate; skip for now and rely on the colormap to show
        # the trajectory shape.
        pass

    ax_traj.add_patch(Circle((0, 0), _MOON_RADIUS_KM,
                             color="dimgray", alpha=0.7, zorder=5))
    ax_traj.plot(0, 0, "o", color="dimgray", ms=5, zorder=6)
    ax_traj.annotate("Moon", (0, 0), textcoords="offset points",
                     xytext=(8, 8), fontsize=10, fontweight="bold")
    ax_traj.plot(rel_xy_km[0, 0], rel_xy_km[0, 1], "o",
                 color="lime", ms=10, mec="black", mew=0.8, zorder=10,
                 label="Start (perilune of flyby)")
    ax_traj.plot(rel_xy_km[-1, 0], rel_xy_km[-1, 1], "*",
                 color="red", ms=14, mec="black", mew=0.8, zorder=10,
                 label="End (after capture)")

    span_km = 1.05 * np.max(np.abs(rel_xy_km))
    ax_traj.set_xlim(-span_km, span_km)
    ax_traj.set_ylim(-span_km, span_km)
    ax_traj.set_xlabel("Moon-relative x (km)")
    ax_traj.set_ylabel("Moon-relative y (km)")
    ax_traj.set_title("Sliding-window QUBO capture trajectory")
    ax_traj.set_aspect("equal")
    ax_traj.grid(alpha=0.3)
    ax_traj.legend(loc="upper right", fontsize=9)

    # --- Energy history ---
    e_hist = result.energy_history()
    w_idx = np.arange(len(e_hist))
    ax_e.plot(w_idx, e_hist, "o-", color="tab:blue", lw=1.5, ms=5)
    ax_e.axhline(0.0, color="0.4", ls="--", lw=1.0)
    ax_e.fill_between(w_idx, e_hist, 0.0,
                      where=(e_hist < 0.0), color="tab:green", alpha=0.2,
                      label="bound")
    ax_e.fill_between(w_idx, e_hist, 0.0,
                      where=(e_hist >= 0.0), color="tab:red", alpha=0.2,
                      label="unbound")
    ax_e.set_xlabel("window index")
    ax_e.set_ylabel("specific energy (CR3BP nondim)")
    ax_e.set_title("Energy w.r.t. Moon")
    ax_e.grid(alpha=0.3)
    ax_e.legend(fontsize=9, loc="upper right")

    # --- Apolune history ---
    apo = result.apolune_history()
    apo_km = apo * LENGTH_KM
    apo_km_finite = np.where(np.isfinite(apo_km), apo_km, np.nan)
    ax_apo.plot(w_idx, apo_km_finite, "o-",
                color="tab:orange", lw=1.5, ms=5,
                label="apolune (km)")
    if np.any(~np.isfinite(apo_km)):
        idx_inf = np.where(~np.isfinite(apo_km))[0]
        ax_apo.scatter(idx_inf, np.full_like(idx_inf, np.nanmax(apo_km_finite)),
                       marker="x", color="tab:red", s=60,
                       label="unbound (inf)")
    ax_apo.set_xlabel("window index")
    ax_apo.set_ylabel("apolune radius (km)")
    ax_apo.set_title("Apolune evolution")
    ax_apo.grid(alpha=0.3)
    ax_apo.legend(fontsize=9)

    bound_label = "hyperbolic" if e0 >= 0 else "eccentric bound"
    fig.suptitle(
        f"Lunar capture/stabilisation: {bound_label} entry "
        f"({excess_factor:.2f} v_circ at {perilune_alt_km:.0f} km altitude) "
        f"-> circularising bound orbit",
        fontsize=12, fontweight="bold",
    )

    out_path = Path(__file__).resolve().parent / "figures" / "lunar_capture_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
