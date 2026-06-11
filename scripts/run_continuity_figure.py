"""Continuity figure: the entire all-binary Earth-Moon mission as a
single continuous trajectory.

Companion to ``run_mission_figure.py``. The three-panel overview
plots each phase in its own natural frame (Earth-inertial for
Phase 1, synodic for Phase 2 and Phase 3), which can hide the fact
that the trajectory is physically continuous: the state at the end
of Phase 1 IS the state at the start of Phase 2, and the state at
the end of Phase 2's useful arc (the perilune) IS the state at the
start of Phase 3.

This figure makes that explicit:

  * One large synodic-frame panel showing Phases 2 and 3 as a single
    continuous curve with a time-progression colormap. The perilune
    appears as a point on the curve, not a discontinuity.
  * Inset (top-right): Phase 1 in Earth-inertial frame, with the
    handoff state marked.

Cumulative ΔV, altitude above Earth/Moon and the per-phase timeline
are reported as text/table values in the companion paper rather than
as a bottom row of small panels.

Output: ``mission_continuity.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.lunar_capture import (
    moon_orbit_apolune_perilune,
    moon_two_body_energy,
)
from qalunar.reference.edelbaum import LENGTH_KM, TIME_S, VELOCITY_M_S

from scripts.run_mission_figure import (
    EARTH_RADIUS_KM,
    MOON_RADIUS_KM,
    R_HEO_KM,
    R_PHASE1_START_KM,
    SPACECRAFT_MASS_KG,
    THRUST_CRUISE_MN,
    THRUST_MAIN_MN,
    T_PHASE2,
    run_mission,
)


_MU = EARTH_MOON_MU
_MOON_POS = np.array([1.0 - _MU, 0.0])
_EARTH_POS = np.array([-_MU, 0.0])


# ---------------------------------------------------------------------------
# Phase 1 spiral generator (Earth-inertial frame, km)
# ---------------------------------------------------------------------------


def _phase1_spiral_inertial(spiral, n_points: int = 30_000) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Generate the Phase-1 Edelbaum spiral in Earth-inertial coordinates.

    Returns (x_km, y_km, t_nondim) sampled uniformly in time.
    """
    gm = 1.0 - _MU
    t = np.linspace(0.0, spiral.tof, n_points)
    sign = np.sign(spiral.v2 - spiral.v1)
    v_t = spiral.v1 + sign * spiral.a_thrust * t
    r_t = gm / v_t ** 2
    dt = t[1] - t[0]
    omega = v_t ** 3 / gm
    theta = np.cumsum(omega) * dt
    x = r_t * np.cos(theta) * LENGTH_KM
    y = r_t * np.sin(theta) * LENGTH_KM
    return x, y, t


def _plot_gradient(ax, x, y, progress, cmap_name, lw=0.7, alpha=0.9,
                   n_subsample=4_000):
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
    return lc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Computing mission...")
    data = run_mission()

    spiral1 = data["spiral1"]
    traj_p2 = data["traj_p2"]
    idx_peri = data["idx_peri"]
    traj_p2_to_peri = traj_p2[: idx_peri + 1]
    traj_p3 = data["traj_p3"]
    times_p3 = data["times_p3"]

    # ---- Build the continuous trajectory ----
    # Phase 2 (truncated at perilune) and Phase 3 share the perilune
    # state. We concatenate them with consistent time axis.
    dt_p2 = T_PHASE2 / (traj_p2.shape[0] - 1)
    times_p2_to_peri = np.linspace(0.0, idx_peri * dt_p2, traj_p2_to_peri.shape[0])
    times_p3_shifted = times_p2_to_peri[-1] + (times_p3 - times_p3[0])

    full_xy = np.concatenate([traj_p2_to_peri[:, :2], traj_p3[:, :2]],
                             axis=0)

    # ---- Plot ----
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "legend.fontsize": 9})

    fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.4, 1.0])

    # Main panel: synodic frame, Phases 2 + 3 continuous
    ax_main = fig.add_subplot(gs[0, :2])
    progress = np.linspace(0.0, 1.0, full_xy.shape[0])
    _plot_gradient(ax_main, full_xy[:, 0], full_xy[:, 1], progress,
                   "viridis", lw=1.0, alpha=0.95)

    # Earth and Moon
    ax_main.plot(*_EARTH_POS, "o", color="tab:blue", ms=12, zorder=8)
    ax_main.annotate("Earth", _EARTH_POS, textcoords="offset points",
                     xytext=(-30, 12), fontsize=11, fontweight="bold")
    ax_main.plot(*_MOON_POS, "o", color="dimgray", ms=8, zorder=8)
    ax_main.annotate("Moon", _MOON_POS, textcoords="offset points",
                     xytext=(8, 12), fontsize=11, fontweight="bold")

    # Phase boundaries on the main panel
    ax_main.plot(traj_p2_to_peri[0, 0], traj_p2_to_peri[0, 1], "o",
                 color="darkviolet", ms=12, mec="white", mew=1.2,
                 zorder=10, label="Phase 2 start (post-injection)")
    ax_main.plot(traj_p2_to_peri[-1, 0], traj_p2_to_peri[-1, 1], "*",
                 color="orange", ms=20, mec="black", mew=1.0,
                 zorder=10, label="Phase 2 → 3 handoff (perilune)")
    ax_main.plot(traj_p3[-1, 0], traj_p3[-1, 1], "X",
                 color="red", ms=14, mec="white", mew=1.2,
                 zorder=10, label="Phase 3 end (captured)")

    ax_main.set_xlim(-0.15, 1.30)
    ax_main.set_ylim(-0.40, 0.95)
    ax_main.set_xlabel("x (synodic, nondim)")
    ax_main.set_ylabel("y (synodic, nondim)")
    ax_main.set_title(
        "Phases 2 + 3: continuous trajectory in the synodic frame "
        "(colour = time progression)"
    )
    ax_main.set_aspect("equal")
    ax_main.grid(alpha=0.3)
    ax_main.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # Phase 1 inset (right): Earth-inertial Edelbaum spiral
    ax_p1 = fig.add_subplot(gs[0, 2])
    x_p1, y_p1, t_p1 = _phase1_spiral_inertial(spiral1)
    prog_p1 = t_p1 / t_p1[-1]
    _plot_gradient(ax_p1, x_p1, y_p1, prog_p1, "Blues",
                   lw=0.7, alpha=0.85, n_subsample=4000)
    ax_p1.add_patch(Circle((0, 0), EARTH_RADIUS_KM, color="tab:blue",
                           alpha=0.6, zorder=5))
    r_start_km = EARTH_RADIUS_KM + R_PHASE1_START_KM
    r_heo_km = R_HEO_KM
    ax_p1.add_patch(Circle((0, 0), r_start_km, fill=False,
                           color="tab:cyan", ls="--", lw=1.0, alpha=0.7,
                           label=f"GEO ({R_PHASE1_START_KM:,.0f} km)"))
    ax_p1.add_patch(Circle((0, 0), r_heo_km, fill=False,
                           color="navy", ls="--", lw=1.0, alpha=0.7,
                           label=f"Parking ({R_HEO_KM/1000:.0f}k km)"))
    ax_p1.plot(x_p1[0], y_p1[0], "o", color="tab:cyan", ms=8,
               mec="black", mew=0.8, zorder=10)
    ax_p1.plot(x_p1[-1], y_p1[-1], "o", color="darkviolet", ms=10,
               mec="white", mew=1.0, zorder=10,
               label="Phase 1 → 2 handoff")
    span = 1.15 * r_heo_km
    ax_p1.set_xlim(-span, span)
    ax_p1.set_ylim(-span, span)
    ax_p1.set_xlabel("Earth-centered x (km)")
    ax_p1.set_ylabel("Earth-centered y (km)")
    ax_p1.set_title("Phase 1: Edelbaum spiral (Earth-inertial)")
    ax_p1.set_aspect("equal")
    ax_p1.grid(alpha=0.3)
    ax_p1.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "All-binary Earth-Moon mission: continuous trajectory across "
        "Phases 1 → 2 → 3",
        fontsize=12, fontweight="bold",
    )

    out_path = Path(__file__).resolve().parent / "figures" / "mission_continuity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
